use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::{Deserialize, Serialize};

// ============================================================
// Configuration
// ============================================================

/// Configuration for the T5-inspired model.
#[derive(Debug, Clone)]
pub struct T5Config {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_encoder_layers: usize,
    pub num_decoder_layers: usize,
    pub d_ff: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub dropout_rate: f64,
}

impl Default for T5Config {
    fn default() -> Self {
        Self {
            d_model: 64,
            num_heads: 4,
            num_encoder_layers: 2,
            num_decoder_layers: 2,
            d_ff: 128,
            vocab_size: 1000,
            max_seq_len: 128,
            dropout_rate: 0.1,
        }
    }
}

// ============================================================
// Sentiment & Trading Signal Types
// ============================================================

/// Sentiment label for financial text.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sentiment {
    Positive,
    Negative,
    Neutral,
}

/// Trading signal derived from sentiment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradingSignal {
    Buy,
    Sell,
    Hold,
}

/// Result of sentiment analysis with confidence score.
#[derive(Debug, Clone)]
pub struct SentimentResult {
    pub sentiment: Sentiment,
    pub confidence: f64,
    pub signal: TradingSignal,
    pub scores: [f64; 3], // [positive, negative, neutral]
}

impl Sentiment {
    /// Convert sentiment to a trading signal.
    pub fn to_signal(self) -> TradingSignal {
        match self {
            Sentiment::Positive => TradingSignal::Buy,
            Sentiment::Negative => TradingSignal::Sell,
            Sentiment::Neutral => TradingSignal::Hold,
        }
    }
}

// ============================================================
// Multi-Head Attention
// ============================================================

/// Multi-head attention mechanism.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub d_model: usize,
    pub d_k: usize,
    pub w_q: Array2<f64>,
    pub w_k: Array2<f64>,
    pub w_v: Array2<f64>,
    pub w_o: Array2<f64>,
}

impl MultiHeadAttention {
    /// Create a new MultiHeadAttention with random weights.
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        let d_k = d_model / num_heads;
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (d_model + d_k) as f64).sqrt();

        let w_q = Array2::from_shape_fn((d_model, d_model), |_| rng.gen::<f64>() * scale - scale / 2.0);
        let w_k = Array2::from_shape_fn((d_model, d_model), |_| rng.gen::<f64>() * scale - scale / 2.0);
        let w_v = Array2::from_shape_fn((d_model, d_model), |_| rng.gen::<f64>() * scale - scale / 2.0);
        let w_o = Array2::from_shape_fn((d_model, d_model), |_| rng.gen::<f64>() * scale - scale / 2.0);

        Self { num_heads, d_model, d_k, w_q, w_k, w_v, w_o }
    }

    /// Compute scaled dot-product attention for a single head.
    pub fn scaled_dot_product_attention(
        query: &Array2<f64>,
        key: &Array2<f64>,
        value: &Array2<f64>,
        mask: bool,
    ) -> Array2<f64> {
        let d_k = query.ncols() as f64;
        let scores = query.dot(&key.t()) / d_k.sqrt();

        // Apply softmax row-wise
        let attention_weights = if mask {
            let mut masked = scores.clone();
            let seq_len = masked.nrows();
            for i in 0..seq_len {
                for j in (i + 1)..masked.ncols() {
                    masked[[i, j]] = f64::NEG_INFINITY;
                }
            }
            softmax_2d(&masked)
        } else {
            softmax_2d(&scores)
        };

        attention_weights.dot(value)
    }

    /// Forward pass: compute multi-head attention.
    pub fn forward(
        &self,
        query: &Array2<f64>,
        key: &Array2<f64>,
        value: &Array2<f64>,
        mask: bool,
    ) -> Array2<f64> {
        let q = query.dot(&self.w_q);
        let k = key.dot(&self.w_k);
        let v = value.dot(&self.w_v);

        let seq_len_q = q.nrows();
        let mut all_heads = Array2::zeros((seq_len_q, self.d_model));

        for h in 0..self.num_heads {
            let start = h * self.d_k;
            let end = start + self.d_k;

            let q_h = q.slice(ndarray::s![.., start..end]).to_owned();
            let k_h = k.slice(ndarray::s![.., start..end]).to_owned();
            let v_h = v.slice(ndarray::s![.., start..end]).to_owned();

            let head_out = Self::scaled_dot_product_attention(&q_h, &k_h, &v_h, mask);

            all_heads.slice_mut(ndarray::s![.., start..end]).assign(&head_out);
        }

        all_heads.dot(&self.w_o)
    }
}

// ============================================================
// Feed-Forward Network
// ============================================================

/// Position-wise feed-forward network with ReLU activation.
#[derive(Debug, Clone)]
pub struct FeedForward {
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
}

impl FeedForward {
    /// Create a new FeedForward network with random weights.
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale1 = (2.0 / (d_model + d_ff) as f64).sqrt();
        let scale2 = (2.0 / (d_ff + d_model) as f64).sqrt();

        Self {
            w1: Array2::from_shape_fn((d_model, d_ff), |_| rng.gen::<f64>() * scale1 - scale1 / 2.0),
            b1: Array1::zeros(d_ff),
            w2: Array2::from_shape_fn((d_ff, d_model), |_| rng.gen::<f64>() * scale2 - scale2 / 2.0),
            b2: Array1::zeros(d_model),
        }
    }

    /// Forward pass with ReLU activation.
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let hidden = x.dot(&self.w1) + &self.b1;
        let relu = hidden.mapv(|v| v.max(0.0));
        relu.dot(&self.w2) + &self.b2
    }
}

// ============================================================
// Layer Normalization
// ============================================================

/// Apply layer normalization along the last axis.
pub fn layer_norm(x: &Array2<f64>) -> Array2<f64> {
    let eps = 1e-6;
    let mean = x.mean_axis(Axis(1)).unwrap();
    let var = x.var_axis(Axis(1), 0.0);

    let mut result = x.clone();
    for i in 0..x.nrows() {
        let std = (var[i] + eps).sqrt();
        for j in 0..x.ncols() {
            result[[i, j]] = (x[[i, j]] - mean[i]) / std;
        }
    }
    result
}

// ============================================================
// Encoder Layer
// ============================================================

/// T5 encoder layer: self-attention + FFN with residual connections.
#[derive(Debug, Clone)]
pub struct EncoderLayer {
    pub self_attention: MultiHeadAttention,
    pub ffn: FeedForward,
}

impl EncoderLayer {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        Self {
            self_attention: MultiHeadAttention::new(d_model, num_heads),
            ffn: FeedForward::new(d_model, d_ff),
        }
    }

    /// Forward pass: self-attention + residual, then FFN + residual.
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        // Self-attention with residual connection and layer norm
        let attn_out = self.self_attention.forward(x, x, x, false);
        let normed1 = layer_norm(&(x + &attn_out));

        // FFN with residual connection and layer norm
        let ffn_out = self.ffn.forward(&normed1);
        layer_norm(&(&normed1 + &ffn_out))
    }
}

// ============================================================
// Decoder Layer
// ============================================================

/// T5 decoder layer: self-attention + cross-attention + FFN.
#[derive(Debug, Clone)]
pub struct DecoderLayer {
    pub self_attention: MultiHeadAttention,
    pub cross_attention: MultiHeadAttention,
    pub ffn: FeedForward,
}

impl DecoderLayer {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        Self {
            self_attention: MultiHeadAttention::new(d_model, num_heads),
            cross_attention: MultiHeadAttention::new(d_model, num_heads),
            ffn: FeedForward::new(d_model, d_ff),
        }
    }

    /// Forward pass through the decoder layer.
    pub fn forward(&self, x: &Array2<f64>, encoder_output: &Array2<f64>) -> Array2<f64> {
        // Masked self-attention with residual
        let self_attn_out = self.self_attention.forward(x, x, x, true);
        let normed1 = layer_norm(&(x + &self_attn_out));

        // Cross-attention with encoder output
        let cross_attn_out = self.cross_attention.forward(&normed1, encoder_output, encoder_output, false);
        let normed2 = layer_norm(&(&normed1 + &cross_attn_out));

        // FFN with residual
        let ffn_out = self.ffn.forward(&normed2);
        layer_norm(&(&normed2 + &ffn_out))
    }
}

// ============================================================
// Span Corruption (Pretraining Objective)
// ============================================================

/// Span corruption for T5 pretraining.
#[derive(Debug)]
pub struct SpanCorruptor {
    pub corruption_rate: f64,
    pub mean_span_length: usize,
}

impl SpanCorruptor {
    pub fn new(corruption_rate: f64, mean_span_length: usize) -> Self {
        Self {
            corruption_rate,
            mean_span_length,
        }
    }

    /// Apply span corruption to a sequence of token IDs.
    /// Returns (corrupted_input, target) as token ID vectors.
    /// Sentinel tokens start from sentinel_start_id.
    pub fn corrupt(&self, tokens: &[u32], sentinel_start_id: u32) -> (Vec<u32>, Vec<u32>) {
        let mut rng = rand::thread_rng();
        let n = tokens.len();
        if n == 0 {
            return (vec![], vec![]);
        }

        let num_to_mask = ((n as f64) * self.corruption_rate).max(1.0) as usize;

        // Generate spans to mask
        let mut masked = vec![false; n];
        let mut total_masked = 0;
        let mut sentinel_id = sentinel_start_id;

        let mut spans: Vec<(usize, usize)> = Vec::new();

        while total_masked < num_to_mask {
            let start = rng.gen_range(0..n);
            let len = rng.gen_range(1..=self.mean_span_length).min(n - start);

            let mut overlap = false;
            for i in start..start + len {
                if masked[i] {
                    overlap = true;
                    break;
                }
            }
            if overlap {
                continue;
            }

            for i in start..start + len {
                masked[i] = true;
            }
            total_masked += len;
            spans.push((start, len));
        }

        spans.sort_by_key(|&(s, _)| s);

        // Build corrupted input and target
        let mut corrupted = Vec::new();
        let mut target = Vec::new();
        let mut i = 0;
        let mut span_idx = 0;
        sentinel_id = sentinel_start_id;

        while i < n {
            if span_idx < spans.len() && i == spans[span_idx].0 {
                // Replace span with sentinel token
                corrupted.push(sentinel_id);
                target.push(sentinel_id);
                let span_len = spans[span_idx].1;
                for j in 0..span_len {
                    target.push(tokens[i + j]);
                }
                sentinel_id += 1;
                i += span_len;
                span_idx += 1;
            } else {
                corrupted.push(tokens[i]);
                i += 1;
            }
        }

        (corrupted, target)
    }
}

// ============================================================
// T5 Model
// ============================================================

/// T5-inspired encoder-decoder model with a sentiment classification head.
#[derive(Debug)]
pub struct T5Model {
    pub config: T5Config,
    pub encoder_layers: Vec<EncoderLayer>,
    pub decoder_layers: Vec<DecoderLayer>,
    pub embedding: Array2<f64>,
    pub sentiment_head: Array2<f64>, // d_model -> 3 (positive, negative, neutral)
}

impl T5Model {
    /// Create a new T5Model with random initialization.
    pub fn new(config: T5Config) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (1.0 / config.d_model as f64).sqrt();

        let encoder_layers = (0..config.num_encoder_layers)
            .map(|_| EncoderLayer::new(config.d_model, config.num_heads, config.d_ff))
            .collect();

        let decoder_layers = (0..config.num_decoder_layers)
            .map(|_| DecoderLayer::new(config.d_model, config.num_heads, config.d_ff))
            .collect();

        let embedding = Array2::from_shape_fn(
            (config.vocab_size, config.d_model),
            |_| rng.gen::<f64>() * scale - scale / 2.0,
        );

        let sentiment_head = Array2::from_shape_fn(
            (config.d_model, 3),
            |_| rng.gen::<f64>() * scale - scale / 2.0,
        );

        Self {
            config,
            encoder_layers,
            decoder_layers,
            embedding,
            sentiment_head,
        }
    }

    /// Embed a sequence of token IDs into a dense representation.
    pub fn embed(&self, token_ids: &[u32]) -> Array2<f64> {
        let seq_len = token_ids.len();
        let mut embedded = Array2::zeros((seq_len, self.config.d_model));
        for (i, &tid) in token_ids.iter().enumerate() {
            let idx = (tid as usize) % self.config.vocab_size;
            embedded.row_mut(i).assign(&self.embedding.row(idx));
        }
        embedded
    }

    /// Run the encoder stack.
    pub fn encode(&self, input: &Array2<f64>) -> Array2<f64> {
        let mut hidden = input.clone();
        for layer in &self.encoder_layers {
            hidden = layer.forward(&hidden);
        }
        hidden
    }

    /// Run the decoder stack.
    pub fn decode(&self, decoder_input: &Array2<f64>, encoder_output: &Array2<f64>) -> Array2<f64> {
        let mut hidden = decoder_input.clone();
        for layer in &self.decoder_layers {
            hidden = layer.forward(&hidden, encoder_output);
        }
        hidden
    }

    /// Classify sentiment from encoder output.
    /// Takes the mean of encoder hidden states and projects to 3 classes.
    pub fn classify_sentiment(&self, encoder_output: &Array2<f64>) -> SentimentResult {
        // Mean pooling over sequence dimension
        let pooled = encoder_output.mean_axis(Axis(0)).unwrap();

        // Project to 3-class logits
        let logits = pooled.dot(&self.sentiment_head);

        // Softmax
        let scores = softmax_1d(&logits);

        let (sentiment, confidence) = if scores[0] > scores[1] && scores[0] > scores[2] {
            (Sentiment::Positive, scores[0])
        } else if scores[1] > scores[2] {
            (Sentiment::Negative, scores[1])
        } else {
            (Sentiment::Neutral, scores[2])
        };

        let signal = sentiment.to_signal();

        SentimentResult {
            sentiment,
            confidence,
            signal,
            scores: [scores[0], scores[1], scores[2]],
        }
    }

    /// Full forward pass for sentiment: embed -> encode -> classify.
    pub fn sentiment_forward(&self, token_ids: &[u32]) -> SentimentResult {
        let embedded = self.embed(token_ids);
        let encoder_output = self.encode(&embedded);
        self.classify_sentiment(&encoder_output)
    }

    /// Full encoder-decoder forward pass.
    pub fn forward(&self, input_ids: &[u32], decoder_ids: &[u32]) -> Array2<f64> {
        let enc_embedded = self.embed(input_ids);
        let encoder_output = self.encode(&enc_embedded);
        let dec_embedded = self.embed(decoder_ids);
        self.decode(&dec_embedded, &encoder_output)
    }
}

// ============================================================
// Simple Tokenizer
// ============================================================

/// Simple whitespace-based tokenizer with a hash function for token IDs.
pub struct SimpleTokenizer {
    pub vocab_size: usize,
}

impl SimpleTokenizer {
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }

    /// Tokenize text into token IDs using a simple hash.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .map(|word| {
                let hash: u32 = word.bytes().fold(0u32, |acc, b| {
                    acc.wrapping_mul(31).wrapping_add(b as u32)
                });
                hash % (self.vocab_size as u32)
            })
            .collect()
    }

    /// Add a task prefix to text before tokenization.
    pub fn tokenize_with_prefix(&self, prefix: &str, text: &str) -> Vec<u32> {
        let full_text = format!("{} {}", prefix, text);
        self.tokenize(&full_text)
    }
}

// ============================================================
// Bybit API Client
// ============================================================

/// Candle (OHLCV) data from Bybit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Bybit API response structure.
#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub list: Vec<Vec<String>>,
}

/// Client for fetching data from Bybit API.
pub struct BybitClient {
    pub base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data from Bybit.
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let resp: BybitResponse = reqwest::blocking::get(&url)?.json()?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", resp.ret_msg);
        }

        let candles = resp
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() >= 6 {
                    Some(Candle {
                        timestamp: row[0].parse().unwrap_or(0),
                        open: row[1].parse().unwrap_or(0.0),
                        high: row[2].parse().unwrap_or(0.0),
                        low: row[3].parse().unwrap_or(0.0),
                        close: row[4].parse().unwrap_or(0.0),
                        volume: row[5].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(candles)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Trading Strategy
// ============================================================

/// Result of a backtest.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub total_pnl: f64,
    pub win_rate: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
}

/// Sentiment-based trading strategy.
pub struct SentimentStrategy {
    pub model: T5Model,
    pub tokenizer: SimpleTokenizer,
    pub buy_threshold: f64,
    pub sell_threshold: f64,
}

impl SentimentStrategy {
    pub fn new(model: T5Model, tokenizer: SimpleTokenizer) -> Self {
        Self {
            model,
            tokenizer,
            buy_threshold: 0.6,
            sell_threshold: 0.4,
        }
    }

    /// Analyze a financial text and produce a trading signal.
    pub fn analyze(&self, text: &str) -> SentimentResult {
        let tokens = self.tokenizer.tokenize_with_prefix("classify financial sentiment:", text);
        self.model.sentiment_forward(&tokens)
    }

    /// Run a backtest using candle data and simulated news headlines.
    pub fn backtest(
        &self,
        candles: &[Candle],
        headlines: &[&str],
    ) -> BacktestResult {
        let mut pnl = 0.0;
        let mut position: f64 = 0.0; // 1.0 = long, -1.0 = short, 0.0 = flat
        let mut entry_price = 0.0;
        let mut total_trades = 0;
        let mut winning_trades = 0;
        let mut losing_trades = 0;
        let mut equity_curve = vec![0.0f64];
        let mut peak_equity = 0.0f64;
        let mut max_drawdown = 0.0f64;
        let mut returns = Vec::new();

        let n = candles.len().min(headlines.len());

        for i in 0..n {
            let result = self.analyze(headlines[i]);
            let price = candles[i].close;

            // Close existing position if signal changes
            if position != 0.0 {
                let trade_pnl = position * (price - entry_price) / entry_price;
                if (result.signal == TradingSignal::Buy && position < 0.0)
                    || (result.signal == TradingSignal::Sell && position > 0.0)
                    || result.signal == TradingSignal::Hold
                {
                    pnl += trade_pnl;
                    returns.push(trade_pnl);
                    if trade_pnl > 0.0 {
                        winning_trades += 1;
                    } else {
                        losing_trades += 1;
                    }
                    total_trades += 1;
                    position = 0.0;
                }
            }

            // Open new position
            if position == 0.0 {
                match result.signal {
                    TradingSignal::Buy if result.confidence > self.buy_threshold => {
                        position = result.confidence;
                        entry_price = price;
                    }
                    TradingSignal::Sell if (1.0 - result.confidence) > (1.0 - self.sell_threshold) => {
                        position = -result.confidence;
                        entry_price = price;
                    }
                    _ => {}
                }
            }

            let equity = pnl + if position != 0.0 {
                position * (price - entry_price) / entry_price
            } else {
                0.0
            };
            equity_curve.push(equity);
            peak_equity = peak_equity.max(equity);
            let drawdown = peak_equity - equity;
            max_drawdown = max_drawdown.max(drawdown);
        }

        let mean_return = if returns.is_empty() {
            0.0
        } else {
            returns.iter().sum::<f64>() / returns.len() as f64
        };
        let std_return = if returns.len() < 2 {
            1.0
        } else {
            let var = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>()
                / (returns.len() - 1) as f64;
            var.sqrt()
        };
        let sharpe_ratio = if std_return > 0.0 {
            mean_return / std_return
        } else {
            0.0
        };

        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };

        BacktestResult {
            total_trades,
            winning_trades,
            losing_trades,
            total_pnl: pnl,
            win_rate,
            max_drawdown,
            sharpe_ratio,
        }
    }
}

// ============================================================
// Utility Functions
// ============================================================

/// Softmax over a 1D array.
pub fn softmax_1d(x: &Array1<f64>) -> Array1<f64> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp = x.mapv(|v| (v - max_val).exp());
    let sum = exp.sum();
    exp / sum
}

/// Row-wise softmax over a 2D array.
pub fn softmax_2d(x: &Array2<f64>) -> Array2<f64> {
    let mut result = x.clone();
    for i in 0..x.nrows() {
        let row = x.row(i);
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for j in 0..x.ncols() {
            result[[i, j]] = (row[j] - max_val).exp();
            sum += result[[i, j]];
        }
        for j in 0..x.ncols() {
            result[[i, j]] /= sum;
        }
    }
    result
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t5_config_default() {
        let config = T5Config::default();
        assert_eq!(config.d_model, 64);
        assert_eq!(config.num_heads, 4);
        assert_eq!(config.num_encoder_layers, 2);
        assert_eq!(config.d_ff, 128);
        assert_eq!(config.vocab_size, 1000);
    }

    #[test]
    fn test_multi_head_attention_output_shape() {
        let mha = MultiHeadAttention::new(64, 4);
        let input = Array2::from_shape_fn((10, 64), |_| rand::thread_rng().gen::<f64>());
        let output = mha.forward(&input, &input, &input, false);
        assert_eq!(output.shape(), &[10, 64]);
    }

    #[test]
    fn test_feed_forward_output_shape() {
        let ffn = FeedForward::new(64, 128);
        let input = Array2::from_shape_fn((5, 64), |_| rand::thread_rng().gen::<f64>());
        let output = ffn.forward(&input);
        assert_eq!(output.shape(), &[5, 64]);
    }

    #[test]
    fn test_encoder_layer_forward() {
        let layer = EncoderLayer::new(64, 4, 128);
        let input = Array2::from_shape_fn((8, 64), |_| rand::thread_rng().gen::<f64>());
        let output = layer.forward(&input);
        assert_eq!(output.shape(), &[8, 64]);
    }

    #[test]
    fn test_decoder_layer_forward() {
        let layer = DecoderLayer::new(64, 4, 128);
        let decoder_input = Array2::from_shape_fn((6, 64), |_| rand::thread_rng().gen::<f64>());
        let encoder_output = Array2::from_shape_fn((8, 64), |_| rand::thread_rng().gen::<f64>());
        let output = layer.forward(&decoder_input, &encoder_output);
        assert_eq!(output.shape(), &[6, 64]);
    }

    #[test]
    fn test_span_corruptor() {
        let corruptor = SpanCorruptor::new(0.15, 3);
        let tokens: Vec<u32> = (0..20).collect();
        let (corrupted, target) = corruptor.corrupt(&tokens, 900);

        // Corrupted should be shorter (spans replaced by sentinels)
        assert!(corrupted.len() <= tokens.len());
        // Target should contain sentinel tokens
        assert!(target.iter().any(|&t| t >= 900));
        // Corrupted should contain sentinel tokens
        assert!(corrupted.iter().any(|&t| t >= 900));
    }

    #[test]
    fn test_t5_model_sentiment_forward() {
        let config = T5Config::default();
        let model = T5Model::new(config);
        let tokens = vec![1, 42, 100, 255, 500];
        let result = model.sentiment_forward(&tokens);

        // Scores should sum to approximately 1.0
        let sum: f64 = result.scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Confidence should be between 0 and 1
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_t5_model_full_forward() {
        let config = T5Config::default();
        let model = T5Model::new(config);
        let input_ids = vec![1, 2, 3, 4, 5];
        let decoder_ids = vec![10, 20, 30];
        let output = model.forward(&input_ids, &decoder_ids);
        assert_eq!(output.shape(), &[3, 64]);
    }

    #[test]
    fn test_simple_tokenizer() {
        let tokenizer = SimpleTokenizer::new(1000);
        let tokens = tokenizer.tokenize("Bitcoin price surges today");
        assert_eq!(tokens.len(), 4);
        assert!(tokens.iter().all(|&t| t < 1000));
    }

    #[test]
    fn test_tokenizer_with_prefix() {
        let tokenizer = SimpleTokenizer::new(1000);
        let tokens = tokenizer.tokenize_with_prefix("classify sentiment:", "good news");
        // "classify" "sentiment:" "good" "news" = 4 tokens
        assert_eq!(tokens.len(), 4);
    }

    #[test]
    fn test_sentiment_to_signal() {
        assert_eq!(Sentiment::Positive.to_signal(), TradingSignal::Buy);
        assert_eq!(Sentiment::Negative.to_signal(), TradingSignal::Sell);
        assert_eq!(Sentiment::Neutral.to_signal(), TradingSignal::Hold);
    }

    #[test]
    fn test_softmax_1d() {
        let x = Array1::from(vec![1.0, 2.0, 3.0]);
        let result = softmax_1d(&x);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Values should be increasing
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_softmax_2d() {
        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = softmax_2d(&x);
        for i in 0..2 {
            let row_sum: f64 = result.row(i).iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_layer_norm() {
        let x = Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();
        let result = layer_norm(&x);
        // Each row should have approximately zero mean
        for i in 0..2 {
            let mean: f64 = result.row(i).iter().sum::<f64>() / 4.0;
            assert!(mean.abs() < 1e-6);
        }
    }

    #[test]
    fn test_backtest_with_simulated_data() {
        let config = T5Config::default();
        let model = T5Model::new(config);
        let tokenizer = SimpleTokenizer::new(1000);
        let strategy = SentimentStrategy::new(model, tokenizer);

        let candles: Vec<Candle> = (0..5)
            .map(|i| Candle {
                timestamp: 1000 + i as u64,
                open: 100.0 + i as f64,
                high: 105.0 + i as f64,
                low: 95.0 + i as f64,
                close: 102.0 + i as f64,
                volume: 1000.0,
            })
            .collect();

        let headlines = vec![
            "Bitcoin surges on institutional adoption",
            "Crypto market crashes amid regulation fears",
            "Markets remain stable with low volatility",
            "Major bank announces crypto services",
            "Economic uncertainty drives safe haven demand",
        ];

        let result = strategy.backtest(&candles, &headlines);
        // Just check it runs without panicking and produces valid output
        assert!(result.win_rate >= 0.0 && result.win_rate <= 1.0);
        assert!(result.max_drawdown >= 0.0);
    }
}
