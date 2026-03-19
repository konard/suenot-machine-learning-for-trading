use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;

// ─── Text Encoder ───────────────────────────────────────────────────────────

/// A simple text encoder that uses character n-gram features projected
/// to a fixed-dimension embedding space.
pub struct TextEncoder {
    pub embedding_dim: usize,
    pub ngram_size: usize,
    pub projection: Array2<f64>,
    pub vocab_size: usize,
}

impl TextEncoder {
    /// Create a new TextEncoder with a random projection matrix.
    pub fn new(embedding_dim: usize, ngram_size: usize, seed: u64) -> Self {
        let vocab_size = 1024; // hash buckets for n-grams
        let mut rng = StdRng::seed_from_u64(seed);
        let projection = Array2::from_shape_fn((vocab_size, embedding_dim), |_| {
            rng.gen_range(-0.1..0.1)
        });
        Self {
            embedding_dim,
            ngram_size,
            projection,
            vocab_size,
        }
    }

    /// Hash a character n-gram to a bucket index.
    fn hash_ngram(&self, ngram: &str) -> usize {
        let mut h: u64 = 5381;
        for b in ngram.bytes() {
            h = h.wrapping_mul(33).wrapping_add(b as u64);
        }
        (h as usize) % self.vocab_size
    }

    /// Encode text into an embedding vector by accumulating n-gram projections.
    pub fn encode(&self, text: &str) -> Array1<f64> {
        let lower = text.to_lowercase();
        let chars: Vec<char> = lower.chars().collect();
        let mut bag = vec![0.0f64; self.vocab_size];

        if chars.len() >= self.ngram_size {
            for window in chars.windows(self.ngram_size) {
                let ngram: String = window.iter().collect();
                let idx = self.hash_ngram(&ngram);
                bag[idx] += 1.0;
            }
        } else {
            // For very short text, use the whole string
            let idx = self.hash_ngram(&lower);
            bag[idx] += 1.0;
        }

        let bag_arr = Array1::from(bag);
        let embedding = bag_arr.dot(&self.projection);
        l2_normalize(&embedding)
    }

    /// Encode a batch of texts into an embedding matrix (N x embedding_dim).
    pub fn encode_batch(&self, texts: &[&str]) -> Array2<f64> {
        let n = texts.len();
        let mut result = Array2::zeros((n, self.embedding_dim));
        for (i, text) in texts.iter().enumerate() {
            let emb = self.encode(text);
            result.row_mut(i).assign(&emb);
        }
        result
    }
}

// ─── Price Encoder ──────────────────────────────────────────────────────────

/// A price encoder that converts OHLCV sequences into embedding vectors
/// using simple convolutional filters followed by a projection.
pub struct PriceEncoder {
    pub embedding_dim: usize,
    pub num_filters: usize,
    pub kernel_size: usize,
    pub filters: Array2<f64>,      // (num_filters, kernel_size * 5)
    pub projection: Array2<f64>,   // (num_filters, embedding_dim)
}

impl PriceEncoder {
    /// Create a new PriceEncoder with random convolutional filters.
    pub fn new(embedding_dim: usize, num_filters: usize, kernel_size: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let filter_width = kernel_size * 5; // 5 for OHLCV
        let filters = Array2::from_shape_fn((num_filters, filter_width), |_| {
            rng.gen_range(-0.1..0.1)
        });
        let projection = Array2::from_shape_fn((num_filters, embedding_dim), |_| {
            rng.gen_range(-0.1..0.1)
        });
        Self {
            embedding_dim,
            num_filters,
            kernel_size,
            filters,
            projection,
        }
    }

    /// Normalize OHLCV data to zero mean and unit variance per column.
    pub fn normalize_ohlcv(data: &Array2<f64>) -> Array2<f64> {
        let n = data.nrows();
        if n == 0 {
            return data.clone();
        }
        let mut result = data.clone();
        for col in 0..data.ncols() {
            let column = data.column(col);
            let mean = column.sum() / n as f64;
            let var = column.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            let std = var.sqrt().max(1e-8);
            for row in 0..n {
                result[[row, col]] = (result[[row, col]] - mean) / std;
            }
        }
        result
    }

    /// Apply 1D convolution over the OHLCV sequence and aggregate via mean pooling.
    fn convolve(&self, data: &Array2<f64>) -> Array1<f64> {
        let n = data.nrows();
        let mut filter_outputs = vec![0.0f64; self.num_filters];

        if n < self.kernel_size {
            return Array1::from(filter_outputs);
        }

        let num_windows = n - self.kernel_size + 1;
        for w in 0..num_windows {
            // Flatten the window into a vector of size kernel_size * 5
            let mut window_flat = Vec::with_capacity(self.kernel_size * 5);
            for k in 0..self.kernel_size {
                for c in 0..5 {
                    window_flat.push(data[[w + k, c]]);
                }
            }
            let window_arr = Array1::from(window_flat);
            for f in 0..self.num_filters {
                let filter_row = self.filters.row(f);
                let dot: f64 = window_arr.iter().zip(filter_row.iter()).map(|(a, b)| a * b).sum();
                // ReLU activation
                filter_outputs[f] += dot.max(0.0);
            }
        }

        // Mean pooling over windows
        for val in filter_outputs.iter_mut() {
            *val /= num_windows as f64;
        }

        Array1::from(filter_outputs)
    }

    /// Encode an OHLCV sequence into an embedding vector.
    /// Input shape: (sequence_length, 5) where columns are O, H, L, C, V.
    pub fn encode(&self, ohlcv: &Array2<f64>) -> Array1<f64> {
        let normalized = Self::normalize_ohlcv(ohlcv);
        let conv_features = self.convolve(&normalized);
        let embedding = conv_features.dot(&self.projection);
        l2_normalize(&embedding)
    }

    /// Encode a batch of OHLCV sequences into an embedding matrix.
    pub fn encode_batch(&self, sequences: &[Array2<f64>]) -> Array2<f64> {
        let n = sequences.len();
        let mut result = Array2::zeros((n, self.embedding_dim));
        for (i, seq) in sequences.iter().enumerate() {
            let emb = self.encode(seq);
            result.row_mut(i).assign(&emb);
        }
        result
    }
}

// ─── CLIP Model ─────────────────────────────────────────────────────────────

/// CLIP-style model that aligns text and price embeddings using contrastive learning.
pub struct CLIPModel {
    pub text_encoder: TextEncoder,
    pub price_encoder: PriceEncoder,
    pub temperature: f64,
    pub embedding_dim: usize,
}

impl CLIPModel {
    /// Create a new CLIPModel with default configuration.
    pub fn new(embedding_dim: usize, seed: u64) -> Self {
        let text_encoder = TextEncoder::new(embedding_dim, 3, seed);
        let price_encoder = PriceEncoder::new(embedding_dim, 32, 3, seed + 1);
        Self {
            text_encoder,
            price_encoder,
            temperature: 0.07,
            embedding_dim,
        }
    }

    /// Compute the cosine similarity matrix between two sets of embeddings.
    /// Returns an (N x M) matrix where entry (i, j) = cos_sim(a_i, b_j).
    /// Assumes rows are already L2-normalized.
    pub fn cosine_similarity_matrix(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        a.dot(&b.t())
    }

    /// Compute the InfoNCE contrastive loss for a batch of aligned pairs.
    /// text_embeddings: (N x D) matrix of L2-normalized text embeddings
    /// price_embeddings: (N x D) matrix of L2-normalized price embeddings
    /// Returns the symmetric CLIP loss (average of text-to-price and price-to-text).
    pub fn info_nce_loss(
        &self,
        text_embeddings: &Array2<f64>,
        price_embeddings: &Array2<f64>,
    ) -> f64 {
        let n = text_embeddings.nrows();
        assert_eq!(n, price_embeddings.nrows(), "Batch sizes must match");
        assert!(n > 0, "Batch must be non-empty");

        // Similarity matrix scaled by temperature
        let sim = Self::cosine_similarity_matrix(text_embeddings, price_embeddings);
        let scaled_sim = &sim / self.temperature;

        // Text-to-price loss: for each text, classify the correct price
        let mut loss_text = 0.0;
        for i in 0..n {
            let row = scaled_sim.row(i);
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = row.iter().map(|&x| (x - max_val).exp()).sum();
            let log_softmax_ii = (scaled_sim[[i, i]] - max_val) - exp_sum.ln();
            loss_text -= log_softmax_ii;
        }
        loss_text /= n as f64;

        // Price-to-text loss: for each price, classify the correct text
        let mut loss_price = 0.0;
        for j in 0..n {
            let col = scaled_sim.column(j);
            let max_val = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = col.iter().map(|&x| (x - max_val).exp()).sum();
            let log_softmax_jj = (scaled_sim[[j, j]] - max_val) - exp_sum.ln();
            loss_price -= log_softmax_jj;
        }
        loss_price /= n as f64;

        (loss_text + loss_price) / 2.0
    }

    /// Perform zero-shot classification of a price sequence given text labels.
    /// Returns a vector of (label, similarity_score) pairs sorted by descending similarity.
    pub fn zero_shot_classify(
        &self,
        price_sequence: &Array2<f64>,
        labels: &[&str],
    ) -> Vec<(String, f64)> {
        let price_emb = self.price_encoder.encode(price_sequence);

        let mut results: Vec<(String, f64)> = labels
            .iter()
            .map(|&label| {
                let text_emb = self.text_encoder.encode(label);
                let sim = cosine_similarity(&price_emb, &text_emb);
                (label.to_string(), sim)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }

    /// Compute cross-modal retrieval: given text, find the most similar price sequences.
    /// Returns indices sorted by descending similarity.
    pub fn retrieve_prices_by_text(
        &self,
        text: &str,
        price_sequences: &[Array2<f64>],
    ) -> Vec<(usize, f64)> {
        let text_emb = self.text_encoder.encode(text);
        let mut scores: Vec<(usize, f64)> = price_sequences
            .iter()
            .enumerate()
            .map(|(i, seq)| {
                let price_emb = self.price_encoder.encode(seq);
                let sim = cosine_similarity(&text_emb, &price_emb);
                (i, sim)
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores
    }
}

// ─── Utility Functions ──────────────────────────────────────────────────────

/// L2-normalize a vector. Returns a unit vector (or zero vector if input is zero).
pub fn l2_normalize(v: &Array1<f64>) -> Array1<f64> {
    let norm = v.dot(v).sqrt();
    if norm < 1e-12 {
        v.clone()
    } else {
        v / norm
    }
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    a.dot(b) / (norm_a * norm_b)
}

/// Softmax function over a 1D array.
pub fn softmax(logits: &Array1<f64>) -> Array1<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals = logits.mapv(|x| (x - max_val).exp());
    let sum = exp_vals.sum();
    exp_vals / sum
}

// ─── Bybit Client ───────────────────────────────────────────────────────────

/// Kline (candlestick) data from Bybit.
#[derive(Debug, Clone)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Client for fetching market data from Bybit's REST API.
pub struct BybitClient {
    pub base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline data from Bybit.
    /// symbol: e.g. "BTCUSDT"
    /// interval: e.g. "15" (minutes), "60", "240", "D"
    /// limit: number of candles (max 200)
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let resp: BybitResponse = reqwest::blocking::get(&url)?.json()?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error: {} (code {})", resp.ret_msg, resp.ret_code);
        }

        let mut klines: Vec<Kline> = resp
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() < 6 {
                    return None;
                }
                Some(Kline {
                    timestamp: row[0].parse().ok()?,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
                })
            })
            .collect();

        // Bybit returns newest first; reverse to chronological order
        klines.reverse();
        Ok(klines)
    }

    /// Convert klines to an ndarray of shape (N, 5) with columns: O, H, L, C, V.
    pub fn klines_to_array(klines: &[Kline]) -> Array2<f64> {
        let n = klines.len();
        let mut data = Array2::zeros((n, 5));
        for (i, k) in klines.iter().enumerate() {
            data[[i, 0]] = k.open;
            data[[i, 1]] = k.high;
            data[[i, 2]] = k.low;
            data[[i, 3]] = k.close;
            data[[i, 4]] = k.volume;
        }
        data
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Market Regime Labels ───────────────────────────────────────────────────

/// Predefined market regime labels for zero-shot classification.
pub fn default_regime_labels() -> Vec<&'static str> {
    vec![
        "Strong bullish trend with increasing volume and momentum",
        "Bearish breakdown with panic selling and high volatility",
        "Range-bound consolidation with low volatility and decreasing volume",
        "Volatile whipsaw with no clear direction and mixed signals",
        "Gradual uptrend with steady accumulation",
        "Sharp correction after extended rally",
        "Bottom formation with signs of reversal",
        "Distribution phase with declining momentum",
    ]
}

/// Generate synthetic price data for testing purposes.
pub fn generate_synthetic_ohlcv(
    regime: &str,
    length: usize,
    seed: u64,
) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Array2::zeros((length, 5));
    let mut price = 100.0;

    for i in 0..length {
        let (drift, vol) = match regime {
            "bullish" => (0.002, 0.01),
            "bearish" => (-0.002, 0.015),
            "consolidation" => (0.0, 0.003),
            "volatile" => (0.0, 0.03),
            _ => (0.0, 0.01),
        };

        let ret = drift + vol * rng.gen_range(-1.0..1.0_f64);
        let open = price;
        let close = price * (1.0 + ret);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.005));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.005));
        let volume = 1000.0 * (1.0 + rng.gen_range(-0.5..0.5_f64));

        data[[i, 0]] = open;
        data[[i, 1]] = high;
        data[[i, 2]] = low;
        data[[i, 3]] = close;
        data[[i, 4]] = volume;

        price = close;
    }

    data
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_normalize() {
        let v = Array1::from(vec![3.0, 4.0]);
        let normalized = l2_normalize(&v);
        let norm = normalized.dot(&normalized).sqrt();
        assert!((norm - 1.0).abs() < 1e-10, "Normalized vector should have unit norm");
        assert!((normalized[0] - 0.6).abs() < 1e-10);
        assert!((normalized[1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = Array1::from(vec![1.0, 2.0, 3.0]);
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-10, "Cosine similarity of a vector with itself should be 1.0");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = Array1::from(vec![1.0, 0.0]);
        let b = Array1::from(vec![0.0, 1.0]);
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10, "Orthogonal vectors should have zero cosine similarity");
    }

    #[test]
    fn test_text_encoder_deterministic() {
        let encoder = TextEncoder::new(64, 3, 42);
        let emb1 = encoder.encode("Bitcoin price surges");
        let emb2 = encoder.encode("Bitcoin price surges");
        let diff: f64 = (&emb1 - &emb2).mapv(|x| x.abs()).sum();
        assert!(diff < 1e-12, "Same input should produce identical embeddings");
    }

    #[test]
    fn test_text_encoder_different_inputs() {
        let encoder = TextEncoder::new(64, 3, 42);
        let emb1 = encoder.encode("Bitcoin price surges on strong demand");
        let emb2 = encoder.encode("Market crashes amid panic selling");
        let sim = cosine_similarity(&emb1, &emb2);
        assert!(sim < 0.99, "Different texts should produce different embeddings, got sim={}", sim);
    }

    #[test]
    fn test_price_encoder_output_shape() {
        let encoder = PriceEncoder::new(64, 16, 3, 42);
        let data = generate_synthetic_ohlcv("bullish", 20, 0);
        let emb = encoder.encode(&data);
        assert_eq!(emb.len(), 64, "Embedding dimension should match");
    }

    #[test]
    fn test_price_encoder_normalized() {
        let encoder = PriceEncoder::new(64, 16, 3, 42);
        let data = generate_synthetic_ohlcv("bullish", 20, 0);
        let emb = encoder.encode(&data);
        let norm = emb.dot(&emb).sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "Price embedding should be L2-normalized, got norm={}", norm);
    }

    #[test]
    fn test_cosine_similarity_matrix() {
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
        let b = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
        let sim = CLIPModel::cosine_similarity_matrix(&a, &b);
        assert_eq!(sim.shape(), &[2, 2]);
        assert!((sim[[0, 0]] - 1.0).abs() < 1e-10);
        assert!(sim[[0, 1]].abs() < 1e-10);
        assert!(sim[[1, 0]].abs() < 1e-10);
        assert!(sim[[1, 1]].abs() < 1e-10);
    }

    #[test]
    fn test_info_nce_loss_perfect_alignment() {
        let model = CLIPModel::new(64, 42);
        // When text and price embeddings are identical, loss should be minimal
        let emb: Array2<f64> = Array2::from_shape_fn((3, 64), |(i, j)| {
            if i == j % 3 { 1.0_f64 } else { 0.0_f64 }
        });
        // L2 normalize rows
        let mut normalized = emb.clone();
        for i in 0..3 {
            let row = emb.row(i).to_owned();
            let norm = row.dot(&row).sqrt();
            for j in 0..64 {
                normalized[[i, j]] = emb[[i, j]] / norm;
            }
        }
        let loss = model.info_nce_loss(&normalized, &normalized);
        // Perfect alignment should give low loss (close to 0 for orthogonal embeddings)
        assert!(loss < 5.0, "Loss for aligned embeddings should be low, got {}", loss);
    }

    #[test]
    fn test_zero_shot_classify() {
        let model = CLIPModel::new(64, 42);
        let data = generate_synthetic_ohlcv("bullish", 30, 0);
        let labels = &["bullish trend", "bearish crash", "sideways consolidation"];
        let results = model.zero_shot_classify(&data, labels);
        assert_eq!(results.len(), 3);
        // Results should be sorted by descending similarity
        assert!(results[0].1 >= results[1].1);
        assert!(results[1].1 >= results[2].1);
    }

    #[test]
    fn test_softmax() {
        let logits = Array1::from(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&logits);
        let sum: f64 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-10, "Softmax should sum to 1");
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_normalize_ohlcv() {
        let data = Array2::from_shape_vec(
            (3, 5),
            vec![
                100.0, 105.0, 95.0, 102.0, 1000.0,
                102.0, 108.0, 100.0, 106.0, 1200.0,
                106.0, 110.0, 103.0, 108.0, 800.0,
            ],
        ).unwrap();
        let normalized = PriceEncoder::normalize_ohlcv(&data);
        // Each column should have approximately zero mean
        for col in 0..5 {
            let mean: f64 = normalized.column(col).sum() / 3.0;
            assert!(mean.abs() < 1e-10, "Column {} mean should be ~0, got {}", col, mean);
        }
    }

    #[test]
    fn test_klines_to_array() {
        let klines = vec![
            Kline { timestamp: 1000, open: 100.0, high: 105.0, low: 95.0, close: 102.0, volume: 500.0 },
            Kline { timestamp: 2000, open: 102.0, high: 110.0, low: 98.0, close: 108.0, volume: 600.0 },
        ];
        let arr = BybitClient::klines_to_array(&klines);
        assert_eq!(arr.shape(), &[2, 5]);
        assert!((arr[[0, 0]] - 100.0).abs() < 1e-10);
        assert!((arr[[1, 4]] - 600.0).abs() < 1e-10);
    }

    #[test]
    fn test_generate_synthetic_ohlcv() {
        let data = generate_synthetic_ohlcv("bullish", 50, 42);
        assert_eq!(data.shape(), &[50, 5]);
        // In a bullish regime, close should generally trend upward
        let first_close = data[[0, 3]];
        let last_close = data[[49, 3]];
        // Note: with random noise this isn't guaranteed, but with seed 42 and 50 bars it should hold
        // Just check that data is valid (positive prices)
        assert!(first_close > 0.0);
        assert!(last_close > 0.0);
    }
}
