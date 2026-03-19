use ndarray::Array1;
use rand::Rng;
use serde::Deserialize;
use std::collections::HashMap;

// ─── Text Encoder ──────────────────────────────────────────────────

/// Encodes raw financial text into a compact feature vector using a
/// hand-crafted sentiment lexicon of positive and negative terms.
///
/// Output vector layout: [positive_count, negative_count,
///   sentiment_score, word_count, avg_word_length]
#[derive(Debug)]
pub struct TextEncoder {
    /// Maps each lexicon term to its sentiment polarity score.
    /// Positive terms have score +1.0, negative terms have -1.0.
    pub lexicon: HashMap<String, f64>,
}

impl TextEncoder {
    /// Create a new `TextEncoder` pre-loaded with ~20 positive and ~20
    /// negative financial terms.
    pub fn new() -> Self {
        let mut lexicon = HashMap::new();

        // Positive financial terms
        let positive_terms = [
            "bullish", "growth", "profit", "revenue", "strong", "beat",
            "exceed", "upgrade", "outperform", "rally", "surge", "gain",
            "positive", "optimistic", "recovery", "dividend", "expansion",
            "momentum", "breakout", "innovative",
        ];
        for term in &positive_terms {
            lexicon.insert(term.to_string(), 1.0);
        }

        // Negative financial terms
        let negative_terms = [
            "bearish", "loss", "decline", "risk", "weak", "miss",
            "downgrade", "underperform", "crash", "plunge", "drop",
            "negative", "pessimistic", "recession", "debt", "contraction",
            "slowdown", "bankruptcy", "default", "volatility",
        ];
        for term in &negative_terms {
            lexicon.insert(term.to_string(), -1.0);
        }

        Self { lexicon }
    }

    /// Encode `text` into a 5-dimensional feature vector:
    /// `[positive_count, negative_count, sentiment_score, word_count, avg_word_length]`
    ///
    /// The sentiment score is defined as (pos - neg) / (pos + neg + 1) so
    /// that it stays in (-1, 1) even when the text is sentiment-neutral.
    pub fn encode(&self, text: &str) -> Vec<f64> {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let word_count = tokens.len() as f64;

        let mut positive_count = 0.0_f64;
        let mut negative_count = 0.0_f64;
        let mut total_char_len = 0usize;

        for token in &tokens {
            let lower = token.to_lowercase();
            // Strip common punctuation from the token before lookup
            let clean: String = lower.chars().filter(|c| c.is_alphabetic()).collect();
            total_char_len += clean.len();

            if let Some(&score) = self.lexicon.get(&clean) {
                if score > 0.0 {
                    positive_count += 1.0;
                } else {
                    negative_count += 1.0;
                }
            }
        }

        let sentiment_score =
            (positive_count - negative_count) / (positive_count + negative_count + 1.0);

        let avg_word_length = if word_count > 0.0 {
            total_char_len as f64 / word_count
        } else {
            0.0
        };

        vec![
            positive_count,
            negative_count,
            sentiment_score,
            word_count,
            avg_word_length,
        ]
    }

    /// Convenience method that returns only the sentiment score component.
    pub fn sentiment_score(&self, text: &str) -> f64 {
        self.encode(text)[2]
    }
}

impl Default for TextEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Numerical Encoder ─────────────────────────────────────────────

/// Projects a raw numerical feature vector into a lower- or
/// higher-dimensional embedding space via a linear transform followed
/// by a tanh activation.
#[derive(Debug)]
pub struct NumericalEncoder {
    /// Weight matrix stored as a flat `Array1` with shape
    /// `output_dim × input_dim` (row-major).
    weights: Array1<f64>,
    bias: f64,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl NumericalEncoder {
    /// Create a new encoder with random small initial weights.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let n = input_dim * output_dim;
        let weights =
            Array1::from_vec((0..n).map(|_| rng.gen_range(-0.1..0.1)).collect());
        Self {
            weights,
            bias: rng.gen_range(-0.1..0.1),
            input_dim,
            output_dim,
        }
    }

    /// Project `features` through the linear transform and apply tanh
    /// element-wise to produce an `output_dim`-length embedding.
    pub fn encode(&self, features: &[f64]) -> Vec<f64> {
        assert_eq!(
            features.len(),
            self.input_dim,
            "Feature length {} does not match input_dim {}",
            features.len(),
            self.input_dim
        );

        let mut output = Vec::with_capacity(self.output_dim);
        for out_idx in 0..self.output_dim {
            let mut sum = self.bias;
            for in_idx in 0..self.input_dim {
                let w_idx = out_idx * self.input_dim + in_idx;
                sum += self.weights[w_idx] * features[in_idx];
            }
            output.push(sum.tanh());
        }
        output
    }

    /// Z-score normalise a slice of values.  Returns a vector of the
    /// same length with mean ≈ 0 and standard deviation ≈ 1.
    /// If the standard deviation is effectively zero the original
    /// values are returned unchanged.
    pub fn z_score_normalize(data: &[f64]) -> Vec<f64> {
        if data.is_empty() {
            return Vec::new();
        }
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / data.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            return data.to_vec();
        }
        data.iter().map(|x| (x - mean) / std_dev).collect()
    }
}

// ─── Visual Feature Extractor ──────────────────────────────────────

/// Extracts technical / visual features from a sequence of OHLCV
/// candlestick bars.
///
/// Each candle is represented as `(open, high, low, close, volume)`.
#[derive(Debug, Default)]
pub struct VisualFeatureExtractor;

impl VisualFeatureExtractor {
    pub fn new() -> Self {
        Self
    }

    /// Compute a 7-dimensional feature vector from `candles`:
    ///
    /// | index | feature            | description                                |
    /// |-------|--------------------|--------------------------------------------|
    /// | 0     | trend_direction    | (last_close − first_close) / first_close   |
    /// | 1     | avg_body_ratio     | mean |close − open| / (high − low)          |
    /// | 2     | avg_upper_shadow   | mean (high − max(open,close)) / (high−low) |
    /// | 3     | avg_lower_shadow   | mean (min(open,close) − low) / (high−low)  |
    /// | 4     | volume_trend       | last_volume / mean_volume                  |
    /// | 5     | bullish_ratio      | fraction of candles where close > open     |
    /// | 6     | volatility         | std-dev of bar-over-bar returns            |
    pub fn extract_features(&self, candles: &[(f64, f64, f64, f64, f64)]) -> Vec<f64> {
        if candles.is_empty() {
            return vec![0.0; 7];
        }

        let n = candles.len();
        let first_close = candles[0].3;
        let last_close = candles[n - 1].3;

        let trend_direction = if first_close != 0.0 {
            (last_close - first_close) / first_close
        } else {
            0.0
        };

        let mut body_ratios = Vec::with_capacity(n);
        let mut upper_shadows = Vec::with_capacity(n);
        let mut lower_shadows = Vec::with_capacity(n);
        let mut bullish_count = 0usize;
        let mut returns = Vec::with_capacity(n.saturating_sub(1));

        for (i, &(open, high, low, close, _volume)) in candles.iter().enumerate() {
            let range = high - low;
            if range > 1e-10 {
                body_ratios.push((close - open).abs() / range);
                upper_shadows
                    .push((high - open.max(close)) / range);
                lower_shadows
                    .push((open.min(close) - low) / range);
            } else {
                body_ratios.push(0.0);
                upper_shadows.push(0.0);
                lower_shadows.push(0.0);
            }

            if close > open {
                bullish_count += 1;
            }

            if i > 0 {
                let prev_close = candles[i - 1].3;
                if prev_close != 0.0 {
                    returns.push((close - prev_close) / prev_close);
                }
            }
        }

        let avg_body_ratio = body_ratios.iter().sum::<f64>() / n as f64;
        let avg_upper_shadow = upper_shadows.iter().sum::<f64>() / n as f64;
        let avg_lower_shadow = lower_shadows.iter().sum::<f64>() / n as f64;

        let total_volume: f64 = candles.iter().map(|c| c.4).sum();
        let avg_volume = total_volume / n as f64;
        let last_volume = candles[n - 1].4;
        let volume_trend = if avg_volume > 1e-10 {
            last_volume / avg_volume
        } else {
            1.0
        };

        let bullish_ratio = bullish_count as f64 / n as f64;

        let volatility = if returns.len() > 1 {
            let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
            let var = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>()
                / returns.len() as f64;
            var.sqrt()
        } else {
            0.0
        };

        vec![
            trend_direction,
            avg_body_ratio,
            avg_upper_shadow,
            avg_lower_shadow,
            volume_trend,
            bullish_ratio,
            volatility,
        ]
    }

    /// Detect bullish (+1) and bearish (-1) engulfing candlestick
    /// patterns for every position in `candles`.  Positions without a
    /// pattern are labeled 0.
    ///
    /// A **bullish engulfing** occurs when the current candle is
    /// bullish (close > open) and its body completely engulfs the
    /// previous bearish candle's body.
    ///
    /// A **bearish engulfing** occurs when the current candle is
    /// bearish (close < open) and its body completely engulfs the
    /// previous bullish candle's body.
    pub fn detect_engulfing(&self, candles: &[(f64, f64, f64, f64, f64)]) -> Vec<i8> {
        let mut signals = vec![0i8; candles.len()];

        for i in 1..candles.len() {
            let (prev_open, _, _, prev_close, _) = candles[i - 1];
            let (curr_open, _, _, curr_close, _) = candles[i];

            let prev_bullish = prev_close > prev_open;
            let curr_bullish = curr_close > curr_open;

            // Bullish engulfing: prev bar was bearish, current bar is bullish
            // and the current body engulfs the previous body.
            if !prev_bullish
                && curr_bullish
                && curr_open <= prev_close
                && curr_close >= prev_open
            {
                signals[i] = 1;
            }
            // Bearish engulfing: prev bar was bullish, current bar is bearish
            // and the current body engulfs the previous body.
            else if prev_bullish
                && !curr_bullish
                && curr_open >= prev_close
                && curr_close <= prev_open
            {
                signals[i] = -1;
            }
        }

        signals
    }
}

// ─── Multimodal Fusion ─────────────────────────────────────────────

/// Fuses text, numerical, and visual embeddings using learned
/// attention weights per modality.
///
/// Attention energies are computed as a dot product of each embedding
/// with a learned weight vector.  A softmax over the three energies
/// produces the final attention weights.
#[derive(Debug)]
pub struct MultimodalFusion {
    /// Attention weight vector for the text modality.
    text_weights: Vec<f64>,
    /// Attention weight vector for the numerical modality.
    numerical_weights: Vec<f64>,
    /// Attention weight vector for the visual modality.
    visual_weights: Vec<f64>,
    pub text_dim: usize,
    pub numerical_dim: usize,
    pub visual_dim: usize,
}

impl MultimodalFusion {
    /// Initialise with random attention weights sized to each modality.
    pub fn new(text_dim: usize, numerical_dim: usize, visual_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let text_weights: Vec<f64> =
            (0..text_dim).map(|_| rng.gen_range(-0.5..0.5)).collect();
        let numerical_weights: Vec<f64> =
            (0..numerical_dim).map(|_| rng.gen_range(-0.5..0.5)).collect();
        let visual_weights: Vec<f64> =
            (0..visual_dim).map(|_| rng.gen_range(-0.5..0.5)).collect();

        Self {
            text_weights,
            numerical_weights,
            visual_weights,
            text_dim,
            numerical_dim,
            visual_dim,
        }
    }

    /// Dot-product of two equal-length slices (zero if lengths differ).
    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| x * y)
            .sum()
    }

    /// Softmax over a slice of energy values.
    fn softmax(energies: [f64; 3]) -> [f64; 3] {
        let max = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: [f64; 3] = [
            (energies[0] - max).exp(),
            (energies[1] - max).exp(),
            (energies[2] - max).exp(),
        ];
        let sum = exps[0] + exps[1] + exps[2];
        [exps[0] / sum, exps[1] / sum, exps[2] / sum]
    }

    /// Compute softmax attention weights over the three modalities.
    ///
    /// Each energy is the dot product of the embedding with its
    /// corresponding learned weight vector, truncated to the shorter
    /// of the two lengths.
    pub fn compute_attention_weights(
        &self,
        text_emb: &[f64],
        num_emb: &[f64],
        vis_emb: &[f64],
    ) -> [f64; 3] {
        let e_text = Self::dot(text_emb, &self.text_weights);
        let e_num = Self::dot(num_emb, &self.numerical_weights);
        let e_vis = Self::dot(vis_emb, &self.visual_weights);
        Self::softmax([e_text, e_num, e_vis])
    }

    /// Fuse embeddings by weighted summation.
    ///
    /// All three embeddings are zero-padded to `max_dim` before the
    /// weighted sum is computed, so the output length equals `max_dim`.
    pub fn fuse(
        &self,
        text_emb: &[f64],
        num_emb: &[f64],
        vis_emb: &[f64],
    ) -> Vec<f64> {
        let (fused, _) = self.fuse_with_weights(text_emb, num_emb, vis_emb);
        fused
    }

    /// Like `fuse`, but also returns the attention weights `[w_text, w_num, w_vis]`.
    pub fn fuse_with_weights(
        &self,
        text_emb: &[f64],
        num_emb: &[f64],
        vis_emb: &[f64],
    ) -> (Vec<f64>, [f64; 3]) {
        let attn = self.compute_attention_weights(text_emb, num_emb, vis_emb);

        let max_dim = text_emb.len().max(num_emb.len()).max(vis_emb.len());
        let mut fused = vec![0.0f64; max_dim];

        for (i, v) in text_emb.iter().enumerate() {
            fused[i] += attn[0] * v;
        }
        for (i, v) in num_emb.iter().enumerate() {
            fused[i] += attn[1] * v;
        }
        for (i, v) in vis_emb.iter().enumerate() {
            fused[i] += attn[2] * v;
        }

        (fused, attn)
    }
}

// ─── Trading Signal ────────────────────────────────────────────────

/// A trading signal produced by `TradingSignalGenerator`.
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Predicted price direction: `true` = up, `false` = down.
    pub direction: bool,
    /// Model confidence in [0.5, 1.0].
    pub confidence: f64,
    /// Softmax attention weights `[text, numerical, visual]`.
    pub modality_weights: [f64; 3],
    /// Fused feature vector used for classification.
    pub fused_features: Vec<f64>,
}

// ─── Trading Signal Generator ──────────────────────────────────────

/// End-to-end pipeline that combines all modality encoders and a
/// logistic-regression classifier to produce a `TradingSignal`.
pub struct TradingSignalGenerator {
    text_encoder: TextEncoder,
    numerical_encoder: NumericalEncoder,
    visual_extractor: VisualFeatureExtractor,
    fusion: MultimodalFusion,
    /// Classifier weights (one per fused feature dimension).
    classifier_weights: Array1<f64>,
    classifier_bias: f64,
}

impl TradingSignalGenerator {
    /// Construct the pipeline with sensible default dimensions:
    /// - text embedding: 5-dim → 8-dim via `NumericalEncoder` logic
    /// - text raw output: 5-dim
    /// - numerical encoding: 5-dim → 8-dim
    /// - visual features: 7-dim
    pub fn new() -> Self {
        let text_encoder = TextEncoder::new();
        let numerical_encoder = NumericalEncoder::new(5, 8);
        let visual_extractor = VisualFeatureExtractor::new();
        // text:8, numerical:8, visual:7  →  max_dim = 8
        let fusion = MultimodalFusion::new(5, 8, 7);

        let mut rng = rand::thread_rng();
        // The fused vector has length = max(5, 8, 7) = 8
        let fused_dim = 8usize;
        let classifier_weights = Array1::from_vec(
            (0..fused_dim)
                .map(|_| rng.gen_range(-0.1..0.1))
                .collect(),
        );
        let classifier_bias = rng.gen_range(-0.1..0.1);

        Self {
            text_encoder,
            numerical_encoder,
            visual_extractor,
            fusion,
            classifier_weights,
            classifier_bias,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Generate a `TradingSignal` from three input modalities:
    ///
    /// - `text`: raw financial news / commentary string
    /// - `market_data`: numerical features (length must be 5)
    /// - `candles`: OHLCV candlestick sequence
    pub fn generate_signal(
        &self,
        text: &str,
        market_data: &[f64],
        candles: &[(f64, f64, f64, f64, f64)],
    ) -> TradingSignal {
        // Encode each modality
        let text_emb = self.text_encoder.encode(text);
        let num_emb = self.numerical_encoder.encode(market_data);
        let vis_emb = self.visual_extractor.extract_features(candles);

        // Fuse with attention
        let (fused, modality_weights) =
            self.fusion.fuse_with_weights(&text_emb, &num_emb, &vis_emb);

        // Logistic-regression classification
        let x = Array1::from_vec(fused.clone());
        let z = self.classifier_weights.dot(&x) + self.classifier_bias;
        let prob = Self::sigmoid(z);
        let direction = prob >= 0.5;
        let confidence = if direction { prob } else { 1.0 - prob };

        TradingSignal {
            direction,
            confidence,
            modality_weights,
            fused_features: fused,
        }
    }
}

impl Default for TradingSignalGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Bybit Client ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
}

#[derive(Debug, Deserialize)]
pub struct KlineResult {
    pub list: Vec<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct OrderbookResult {
    pub b: Vec<Vec<String>>, // bids: [price, size]
    pub a: Vec<Vec<String>>, // asks: [price, size]
}

/// A parsed kline bar.
#[derive(Debug, Clone)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Async client for the Bybit V5 public market-data API.
pub struct BybitClient {
    base_url: String,
    client: reqwest::Client,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Fetch kline (candlestick) data.
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> anyhow::Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );
        let resp: BybitResponse<KlineResult> =
            self.client.get(&url).send().await?.json().await?;

        let mut klines = Vec::new();
        for item in &resp.result.list {
            if item.len() >= 6 {
                klines.push(Kline {
                    timestamp: item[0].parse().unwrap_or(0),
                    open: item[1].parse().unwrap_or(0.0),
                    high: item[2].parse().unwrap_or(0.0),
                    low: item[3].parse().unwrap_or(0.0),
                    close: item[4].parse().unwrap_or(0.0),
                    volume: item[5].parse().unwrap_or(0.0),
                });
            }
        }
        klines.reverse(); // Bybit returns newest first
        Ok(klines)
    }

    /// Fetch order book snapshot.
    pub async fn get_orderbook(
        &self,
        symbol: &str,
        limit: u32,
    ) -> anyhow::Result<(Vec<(f64, f64)>, Vec<(f64, f64)>)> {
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url, symbol, limit
        );
        let resp: BybitResponse<OrderbookResult> =
            self.client.get(&url).send().await?.json().await?;

        let bids: Vec<(f64, f64)> = resp
            .result
            .b
            .iter()
            .filter_map(|entry| {
                if entry.len() >= 2 {
                    Some((
                        entry[0].parse().unwrap_or(0.0),
                        entry[1].parse().unwrap_or(0.0),
                    ))
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<(f64, f64)> = resp
            .result
            .a
            .iter()
            .filter_map(|entry| {
                if entry.len() >= 2 {
                    Some((
                        entry[0].parse().unwrap_or(0.0),
                        entry[1].parse().unwrap_or(0.0),
                    ))
                } else {
                    None
                }
            })
            .collect();

        Ok((bids, asks))
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Synthetic Data Generation ─────────────────────────────────────

/// Generate `n` synthetic OHLCV candles starting at `start_price` using
/// a random-walk price model.
pub fn generate_synthetic_candles(
    n: usize,
    start_price: f64,
) -> Vec<(f64, f64, f64, f64, f64)> {
    let mut rng = rand::thread_rng();
    let mut price = start_price;
    let mut candles = Vec::with_capacity(n);

    for _ in 0..n {
        let open = price;
        let change = rng.gen_range(-0.02..0.02) * price;
        let close = (price + change).max(1.0);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100.0..10_000.0);
        candles.push((open, high, low, close, volume));
        price = close;
    }
    candles
}

/// Return a set of sample financial news strings spanning bullish,
/// bearish, and neutral sentiment.
pub fn generate_sample_texts() -> Vec<&'static str> {
    vec![
        "Company reports strong revenue growth and profit beat, stock rally expected.",
        "Analysts upgrade outlook as earnings exceed expectations, bullish momentum builds.",
        "Market shows recovery signals with expansion in key sectors and positive dividend news.",
        "Recession fears grow as debt levels rise and economic contraction deepens.",
        "Stock plunge continues amid bankruptcy concerns and bearish investor sentiment.",
        "Volatility spikes as default risk and credit downgrade weigh on market.",
        "Breakout pattern emerges with innovative product launch driving outperform ratings.",
        "Mixed signals as some sectors show growth while others face slowdown and risk.",
    ]
}

/// Generate `n` synthetic market-data rows.
///
/// Each row contains five features:
/// `[return, volume_ratio, volatility, momentum, spread]`
pub fn generate_synthetic_market_data(n: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| {
            vec![
                rng.gen_range(-0.05..0.05),  // return
                rng.gen_range(0.5..2.0),      // volume_ratio
                rng.gen_range(0.001..0.05),   // volatility
                rng.gen_range(-0.1..0.1),     // momentum
                rng.gen_range(0.0001..0.005), // spread
            ]
        })
        .collect()
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_encoder_positive() {
        let enc = TextEncoder::new();
        let features = enc.encode("bullish growth profit revenue rally surge momentum");
        // positive_count >= 7 (all terms are positive)
        assert!(features[0] >= 7.0, "expected >= 7 positive hits, got {}", features[0]);
        // sentiment_score should be positive
        assert!(features[2] > 0.0, "expected positive sentiment score, got {}", features[2]);
    }

    #[test]
    fn test_text_encoder_negative() {
        let enc = TextEncoder::new();
        let features = enc.encode("bearish loss decline crash recession bankruptcy default");
        // negative_count >= 7
        assert!(features[1] >= 7.0, "expected >= 7 negative hits, got {}", features[1]);
        // sentiment_score should be negative
        assert!(features[2] < 0.0, "expected negative sentiment score, got {}", features[2]);
    }

    #[test]
    fn test_text_encoder_neutral() {
        let enc = TextEncoder::new();
        let features = enc.encode("the market closed today at its usual level");
        // No lexicon hits → both counts are 0
        assert_eq!(features[0], 0.0);
        assert_eq!(features[1], 0.0);
        // Sentiment score for no hits: (0 - 0) / (0 + 0 + 1) = 0
        assert!((features[2] - 0.0).abs() < 1e-9);
        // Word count matches token count
        assert_eq!(features[3], 8.0);
    }

    #[test]
    fn test_numerical_encoder_dimensions() {
        let enc = NumericalEncoder::new(5, 8);
        let input = vec![0.1, -0.2, 0.5, 1.0, -0.3];
        let output = enc.encode(&input);
        assert_eq!(output.len(), 8, "output should have 8 dimensions");
        // tanh output must be in (-1, 1)
        for &v in &output {
            assert!(v > -1.0 && v < 1.0, "tanh value out of range: {}", v);
        }
    }

    #[test]
    fn test_z_score_normalize() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let norm = NumericalEncoder::z_score_normalize(&data);
        assert_eq!(norm.len(), data.len());
        let mean: f64 = norm.iter().sum::<f64>() / norm.len() as f64;
        assert!(mean.abs() < 1e-9, "normalised mean should be ~0, got {}", mean);
        let var: f64 =
            norm.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / norm.len() as f64;
        assert!((var - 1.0).abs() < 1e-6, "normalised variance should be ~1, got {}", var);
    }

    #[test]
    fn test_visual_feature_extraction() {
        let candles = generate_synthetic_candles(30, 50_000.0);
        let extractor = VisualFeatureExtractor::new();
        let features = extractor.extract_features(&candles);
        assert_eq!(features.len(), 7, "expected 7 visual features");
        // volume_trend and bullish_ratio must be non-negative
        assert!(features[4] >= 0.0, "volume_trend must be >= 0");
        assert!(features[5] >= 0.0 && features[5] <= 1.0, "bullish_ratio must be in [0,1]");
        assert!(features[6] >= 0.0, "volatility must be >= 0");
    }

    #[test]
    fn test_engulfing_detection() {
        let extractor = VisualFeatureExtractor::new();
        // Craft a known bullish-engulfing pattern:
        // Bar 0: bearish (open=100, close=95)
        // Bar 1: bullish (open=94, close=102) — engulfs bar 0
        let candles = vec![
            (100.0, 101.0, 94.0, 95.0, 500.0),
            (94.0, 103.0, 93.0, 102.0, 800.0),
        ];
        let signals = extractor.detect_engulfing(&candles);
        assert_eq!(signals.len(), 2);
        assert_eq!(signals[0], 0, "first bar cannot have a pattern");
        assert_eq!(signals[1], 1, "expected bullish engulfing at bar 1");
    }

    #[test]
    fn test_fusion_attention_weights() {
        let fusion = MultimodalFusion::new(5, 8, 7);
        let text_emb = vec![0.1; 5];
        let num_emb = vec![0.2; 8];
        let vis_emb = vec![0.3; 7];
        let weights = fusion.compute_attention_weights(&text_emb, &num_emb, &vis_emb);
        // Softmax weights must sum to 1
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "attention weights must sum to 1, got {}", sum);
        // Each weight must be in (0, 1)
        for w in &weights {
            assert!(*w > 0.0 && *w < 1.0);
        }
    }

    #[test]
    fn test_fusion_output_dimension() {
        let fusion = MultimodalFusion::new(5, 8, 7);
        let text_emb = vec![0.1f64; 5];
        let num_emb = vec![0.2f64; 8];
        let vis_emb = vec![0.3f64; 7];
        let fused = fusion.fuse(&text_emb, &num_emb, &vis_emb);
        // max_dim = max(5, 8, 7) = 8
        assert_eq!(fused.len(), 8, "fused vector length should equal max modality dim");
    }

    #[test]
    fn test_trading_signal_generation() {
        let generator = TradingSignalGenerator::new();
        let candles = generate_synthetic_candles(20, 50_000.0);
        let market_data = vec![0.01, 1.2, 0.02, 0.05, 0.001];
        let text = "Strong growth and bullish momentum drive profit rally.";

        let signal = generator.generate_signal(text, &market_data, &candles);

        assert!(
            signal.confidence >= 0.5 && signal.confidence <= 1.0,
            "confidence out of range: {}",
            signal.confidence
        );
        let weight_sum: f64 = signal.modality_weights.iter().sum();
        assert!(
            (weight_sum - 1.0).abs() < 1e-9,
            "modality weights must sum to 1, got {}",
            weight_sum
        );
        assert_eq!(signal.fused_features.len(), 8);
    }

    #[test]
    fn test_sample_texts() {
        let texts = generate_sample_texts();
        assert!(texts.len() >= 6, "expected at least 6 sample texts");
        let enc = TextEncoder::new();
        for text in &texts {
            let features = enc.encode(text);
            assert_eq!(features.len(), 5);
            // Each text should have at least one word
            assert!(features[3] > 0.0, "word_count should be > 0 for: {}", text);
        }
    }

    #[test]
    fn test_synthetic_candles() {
        let candles = generate_synthetic_candles(50, 30_000.0);
        assert_eq!(candles.len(), 50);
        for &(open, high, low, close, volume) in &candles {
            assert!(high >= open, "high must be >= open");
            assert!(high >= close, "high must be >= close");
            assert!(low <= open, "low must be <= open");
            assert!(low <= close, "low must be <= close");
            assert!(volume > 0.0, "volume must be positive");
        }
    }
}
