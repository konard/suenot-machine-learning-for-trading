//! Contrastive Self-Supervised Pretraining for Financial Time Series
//!
//! This crate implements SimCLR, MoCo, and BYOL-style contrastive learning
//! frameworks adapted for financial time series data. It includes data
//! augmentation pipelines, encoder networks, projection heads, contrastive
//! losses, and a Bybit API client for fetching real market data.

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

// ============================================================
// Data Augmentation Pipeline
// ============================================================

/// Configuration for time series augmentation.
#[derive(Debug, Clone)]
pub struct AugmentationConfig {
    /// Probability of applying time masking.
    pub time_mask_prob: f64,
    /// Maximum fraction of the window to mask.
    pub time_mask_max_frac: f64,
    /// Probability of applying magnitude scaling.
    pub scale_prob: f64,
    /// Range for scale factor: [1 - scale_range, 1 + scale_range].
    pub scale_range: f64,
    /// Probability of applying Gaussian jitter.
    pub jitter_prob: f64,
    /// Standard deviation multiplier for jitter (relative to data std).
    pub jitter_std_factor: f64,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            time_mask_prob: 0.5,
            time_mask_max_frac: 0.15,
            scale_prob: 0.5,
            scale_range: 0.2,
            jitter_prob: 0.5,
            jitter_std_factor: 0.01,
        }
    }
}

/// Augmenter that applies stochastic transformations to time series data.
pub struct Augmenter {
    pub config: AugmentationConfig,
}

impl Augmenter {
    pub fn new(config: AugmentationConfig) -> Self {
        Self { config }
    }

    /// Apply time masking: replace a contiguous segment with zeros.
    pub fn time_mask(&self, data: &Array1<f64>) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let mut result = data.clone();
        if rng.gen::<f64>() > self.config.time_mask_prob {
            return result;
        }
        let len = data.len();
        if len == 0 {
            return result;
        }
        let mask_len = (len as f64 * self.config.time_mask_max_frac * rng.gen::<f64>()) as usize;
        let mask_len = mask_len.max(1).min(len);
        let start = rng.gen_range(0..=(len - mask_len));
        for i in start..(start + mask_len) {
            result[i] = 0.0;
        }
        result
    }

    /// Apply magnitude scaling: multiply all values by a random factor.
    pub fn magnitude_scale(&self, data: &Array1<f64>) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() > self.config.scale_prob {
            return data.clone();
        }
        let scale =
            1.0 + rng.gen_range(-self.config.scale_range..self.config.scale_range);
        data * scale
    }

    /// Apply Gaussian jitter: add small random noise.
    pub fn gaussian_jitter(&self, data: &Array1<f64>) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() > self.config.jitter_prob {
            return data.clone();
        }
        let std = data_std(data) * self.config.jitter_std_factor;
        let noise: Array1<f64> = Array1::from_shape_fn(data.len(), |_| {
            let u1: f64 = rng.gen::<f64>().max(1e-10);
            let u2: f64 = rng.gen::<f64>();
            std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        });
        data + &noise
    }

    /// Apply the full augmentation pipeline (composition of all transforms).
    pub fn augment(&self, data: &Array1<f64>) -> Array1<f64> {
        let x = self.time_mask(data);
        let x = self.magnitude_scale(&x);
        self.gaussian_jitter(&x)
    }

    /// Create a positive pair from a single sample.
    pub fn create_positive_pair(&self, data: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        (self.augment(data), self.augment(data))
    }
}

/// Compute standard deviation of an array.
fn data_std(data: &Array1<f64>) -> f64 {
    let n = data.len() as f64;
    if n <= 1.0 {
        return 0.0;
    }
    let mean = data.sum() / n;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    variance.sqrt()
}

// ============================================================
// Encoder Network
// ============================================================

/// Simple feedforward encoder that maps time series windows to fixed-size representations.
///
/// Architecture: Input -> Linear -> ReLU -> Linear -> representation
#[derive(Debug, Clone)]
pub struct Encoder {
    /// Weights for first linear layer: (hidden_dim, input_dim)
    pub w1: Array2<f64>,
    /// Bias for first linear layer: (hidden_dim,)
    pub b1: Array1<f64>,
    /// Weights for second linear layer: (repr_dim, hidden_dim)
    pub w2: Array2<f64>,
    /// Bias for second linear layer: (repr_dim,)
    pub b2: Array1<f64>,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub repr_dim: usize,
}

impl Encoder {
    /// Create a new encoder with random weights (Xavier initialization).
    pub fn new(input_dim: usize, hidden_dim: usize, repr_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale1 = (2.0 / (input_dim + hidden_dim) as f64).sqrt();
        let scale2 = (2.0 / (hidden_dim + repr_dim) as f64).sqrt();

        let w1 = Array2::from_shape_fn((hidden_dim, input_dim), |_| {
            rng.gen_range(-scale1..scale1)
        });
        let b1 = Array1::zeros(hidden_dim);
        let w2 = Array2::from_shape_fn((repr_dim, hidden_dim), |_| {
            rng.gen_range(-scale2..scale2)
        });
        let b2 = Array1::zeros(repr_dim);

        Self {
            w1,
            b1,
            w2,
            b2,
            input_dim,
            hidden_dim,
            repr_dim,
        }
    }

    /// Forward pass: encode an input vector to a representation.
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        // First layer: h = ReLU(W1 * x + b1)
        let h = self.w1.dot(x) + &self.b1;
        let h = h.mapv(|v| v.max(0.0)); // ReLU

        // Second layer: out = W2 * h + b2
        self.w2.dot(&h) + &self.b2
    }

    /// Encode a batch of inputs.
    pub fn encode_batch(&self, batch: &[Array1<f64>]) -> Vec<Array1<f64>> {
        batch.iter().map(|x| self.forward(x)).collect()
    }
}

// ============================================================
// Projection Head
// ============================================================

/// Two-layer MLP projection head for mapping representations to contrastive space.
///
/// Architecture: Input -> Linear -> ReLU -> Linear -> projection
#[derive(Debug, Clone)]
pub struct ProjectionHead {
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
}

impl ProjectionHead {
    /// Create a new projection head with random weights.
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale1 = (2.0 / (input_dim + hidden_dim) as f64).sqrt();
        let scale2 = (2.0 / (hidden_dim + output_dim) as f64).sqrt();

        Self {
            w1: Array2::from_shape_fn((hidden_dim, input_dim), |_| {
                rng.gen_range(-scale1..scale1)
            }),
            b1: Array1::zeros(hidden_dim),
            w2: Array2::from_shape_fn((output_dim, hidden_dim), |_| {
                rng.gen_range(-scale2..scale2)
            }),
            b2: Array1::zeros(output_dim),
        }
    }

    /// Forward pass through the projection head.
    pub fn forward(&self, h: &Array1<f64>) -> Array1<f64> {
        let hidden = self.w1.dot(h) + &self.b1;
        let hidden = hidden.mapv(|v| v.max(0.0)); // ReLU
        self.w2.dot(&hidden) + &self.b2
    }
}

// ============================================================
// NT-Xent Contrastive Loss
// ============================================================

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let dot = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// L2-normalize a vector.
pub fn l2_normalize(v: &Array1<f64>) -> Array1<f64> {
    let norm = v.dot(v).sqrt();
    if norm < 1e-10 {
        return v.clone();
    }
    v / norm
}

/// Compute the NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss
/// for a batch of positive pairs.
///
/// `projections` contains pairs: [(z_1a, z_1b), (z_2a, z_2b), ...] where
/// (z_ia, z_ib) are projections of the two augmented views of sample i.
///
/// Returns the average NT-Xent loss over all positive pairs.
pub fn nt_xent_loss(projections: &[(Array1<f64>, Array1<f64>)], temperature: f64) -> f64 {
    let n = projections.len();
    if n == 0 {
        return 0.0;
    }

    // Flatten into a single list: [z_1a, z_1b, z_2a, z_2b, ...]
    let mut all_z: Vec<Array1<f64>> = Vec::with_capacity(2 * n);
    for (za, zb) in projections {
        all_z.push(za.clone());
        all_z.push(zb.clone());
    }
    let total = all_z.len(); // 2N

    // Compute pairwise similarity matrix
    let mut sim_matrix = Array2::zeros((total, total));
    for i in 0..total {
        for j in 0..total {
            sim_matrix[[i, j]] = cosine_similarity(&all_z[i], &all_z[j]);
        }
    }

    // Compute loss for each element
    let mut total_loss = 0.0;
    for i in 0..total {
        // Positive partner: if i is even (2k), partner is i+1 (2k+1); if odd (2k+1), partner is i-1 (2k)
        let j = if i % 2 == 0 { i + 1 } else { i - 1 };

        let pos_sim = sim_matrix[[i, j]] / temperature;

        // Sum of exp(sim/tau) over all k != i
        let mut denom = 0.0;
        for k in 0..total {
            if k != i {
                denom += (sim_matrix[[i, k]] / temperature).exp();
            }
        }

        total_loss += -(pos_sim - denom.ln());
    }

    total_loss / total as f64
}

// ============================================================
// MoCo Momentum Encoder
// ============================================================

/// Momentum Contrast (MoCo) encoder with a queue of negative keys.
pub struct MomentumEncoder {
    /// The query encoder (updated by gradient descent).
    pub query_encoder: Encoder,
    /// The key encoder (updated via momentum).
    pub key_encoder: Encoder,
    /// The projection head for queries.
    pub query_projector: ProjectionHead,
    /// The projection head for keys.
    pub key_projector: ProjectionHead,
    /// Momentum coefficient for EMA update.
    pub momentum: f64,
    /// Queue of past key projections.
    pub queue: Vec<Array1<f64>>,
    /// Maximum queue size.
    pub queue_size: usize,
}

impl MomentumEncoder {
    /// Create a new MoCo encoder.
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        repr_dim: usize,
        proj_dim: usize,
        momentum: f64,
        queue_size: usize,
    ) -> Self {
        let query_encoder = Encoder::new(input_dim, hidden_dim, repr_dim);
        let key_encoder = query_encoder.clone();
        let query_projector = ProjectionHead::new(repr_dim, repr_dim, proj_dim);
        let key_projector = query_projector.clone();

        Self {
            query_encoder,
            key_encoder,
            query_projector,
            key_projector,
            momentum,
            queue: Vec::new(),
            queue_size,
        }
    }

    /// Update key encoder parameters using exponential moving average.
    pub fn momentum_update(&mut self) {
        let m = self.momentum;
        // Update key encoder weights from query encoder
        self.key_encoder.w1 = &self.key_encoder.w1 * m + &self.query_encoder.w1 * (1.0 - m);
        self.key_encoder.b1 = &self.key_encoder.b1 * m + &self.query_encoder.b1 * (1.0 - m);
        self.key_encoder.w2 = &self.key_encoder.w2 * m + &self.query_encoder.w2 * (1.0 - m);
        self.key_encoder.b2 = &self.key_encoder.b2 * m + &self.query_encoder.b2 * (1.0 - m);

        // Update key projector weights
        self.key_projector.w1 =
            &self.key_projector.w1 * m + &self.query_projector.w1 * (1.0 - m);
        self.key_projector.b1 =
            &self.key_projector.b1 * m + &self.query_projector.b1 * (1.0 - m);
        self.key_projector.w2 =
            &self.key_projector.w2 * m + &self.query_projector.w2 * (1.0 - m);
        self.key_projector.b2 =
            &self.key_projector.b2 * m + &self.query_projector.b2 * (1.0 - m);
    }

    /// Encode a query sample.
    pub fn encode_query(&self, x: &Array1<f64>) -> Array1<f64> {
        let h = self.query_encoder.forward(x);
        self.query_projector.forward(&h)
    }

    /// Encode a key sample (no gradient).
    pub fn encode_key(&self, x: &Array1<f64>) -> Array1<f64> {
        let h = self.key_encoder.forward(x);
        self.key_projector.forward(&h)
    }

    /// Enqueue new keys and dequeue old ones if over capacity.
    pub fn update_queue(&mut self, keys: &[Array1<f64>]) {
        for k in keys {
            self.queue.push(k.clone());
        }
        while self.queue.len() > self.queue_size {
            self.queue.remove(0);
        }
    }

    /// Compute InfoNCE loss for a batch of query-key pairs.
    pub fn info_nce_loss(
        &self,
        queries: &[Array1<f64>],
        positive_keys: &[Array1<f64>],
        temperature: f64,
    ) -> f64 {
        let mut total_loss = 0.0;
        let n = queries.len();

        for i in 0..n {
            let q = &queries[i];
            let k_pos = &positive_keys[i];

            let pos_sim = cosine_similarity(q, k_pos) / temperature;

            let mut denom = pos_sim.exp();
            for neg_key in &self.queue {
                let neg_sim = cosine_similarity(q, neg_key) / temperature;
                denom += neg_sim.exp();
            }

            total_loss += -(pos_sim - denom.ln());
        }

        if n > 0 {
            total_loss / n as f64
        } else {
            0.0
        }
    }
}

// ============================================================
// Linear Probe for Downstream Evaluation
// ============================================================

/// Simple linear classifier for evaluating representation quality.
pub struct LinearProbe {
    /// Weight matrix: (num_classes, repr_dim)
    pub weights: Array2<f64>,
    /// Bias: (num_classes,)
    pub bias: Array1<f64>,
    pub num_classes: usize,
    pub repr_dim: usize,
    pub learning_rate: f64,
}

impl LinearProbe {
    /// Create a new linear probe.
    pub fn new(repr_dim: usize, num_classes: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (repr_dim + num_classes) as f64).sqrt();
        Self {
            weights: Array2::from_shape_fn((num_classes, repr_dim), |_| {
                rng.gen_range(-scale..scale)
            }),
            bias: Array1::zeros(num_classes),
            num_classes,
            repr_dim,
            learning_rate,
        }
    }

    /// Compute softmax probabilities.
    pub fn predict_proba(&self, representation: &Array1<f64>) -> Array1<f64> {
        let logits = self.weights.dot(representation) + &self.bias;
        softmax(&logits)
    }

    /// Predict the class label.
    pub fn predict(&self, representation: &Array1<f64>) -> usize {
        let probs = self.predict_proba(representation);
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Train one step using a single sample with cross-entropy loss.
    /// Returns the loss value.
    pub fn train_step(&mut self, representation: &Array1<f64>, label: usize) -> f64 {
        let logits = self.weights.dot(representation) + &self.bias;
        let probs = softmax(&logits);

        // Cross-entropy loss
        let loss = -(probs[label].max(1e-10)).ln();

        // Gradient: d_loss/d_logits = probs - one_hot(label)
        let mut grad_logits = probs.clone();
        grad_logits[label] -= 1.0;

        // Update weights: W -= lr * grad_logits outer representation
        for c in 0..self.num_classes {
            for d in 0..self.repr_dim {
                self.weights[[c, d]] -=
                    self.learning_rate * grad_logits[c] * representation[d];
            }
            self.bias[c] -= self.learning_rate * grad_logits[c];
        }

        loss
    }

    /// Evaluate accuracy on a labeled dataset.
    pub fn evaluate(
        &self,
        representations: &[Array1<f64>],
        labels: &[usize],
    ) -> f64 {
        let correct = representations
            .iter()
            .zip(labels.iter())
            .filter(|(r, &l)| self.predict(r) == l)
            .count();
        correct as f64 / representations.len() as f64
    }
}

/// Compute softmax of a vector.
fn softmax(logits: &Array1<f64>) -> Array1<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Array1<f64> = logits.mapv(|x| (x - max_val).exp());
    let sum: f64 = exp_vals.sum();
    exp_vals / sum
}

// ============================================================
// Bybit API Client
// ============================================================

/// Response structure for Bybit V5 kline API.
#[derive(Debug, Deserialize, Serialize)]
pub struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i64,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitKlineResult,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct BybitKlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// OHLCV candle data.
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Bybit API client for fetching market data.
pub struct BybitClient {
    pub base_url: String,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data from Bybit V5 API.
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let client = reqwest::Client::new();
        let resp: BybitKlineResponse = client.get(&url).send().await?.json().await?;

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

    /// Fetch klines using blocking (synchronous) client.
    pub fn fetch_klines_blocking(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let client = reqwest::blocking::Client::new();
        let resp: BybitKlineResponse = client.get(&url).send()?.json()?;

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

// ============================================================
// Utility Functions
// ============================================================

/// Convert candles to close price array.
pub fn candles_to_close_prices(candles: &[Candle]) -> Array1<f64> {
    Array1::from_vec(candles.iter().map(|c| c.close).collect())
}

/// Create overlapping windows from a 1D array.
pub fn create_windows(data: &Array1<f64>, window_size: usize, stride: usize) -> Vec<Array1<f64>> {
    let mut windows = Vec::new();
    let mut start = 0;
    while start + window_size <= data.len() {
        let window = data.slice(ndarray::s![start..start + window_size]).to_owned();
        windows.push(window);
        start += stride;
    }
    windows
}

/// Z-score normalize a window.
pub fn zscore_normalize(data: &Array1<f64>) -> Array1<f64> {
    let mean = data.sum() / data.len() as f64;
    let std = data_std(data);
    if std < 1e-10 {
        return data - mean;
    }
    (data - mean) / std
}

/// Run a full SimCLR-style contrastive step on a batch of windows.
/// Returns the NT-Xent loss for the batch.
pub fn simclr_step(
    windows: &[Array1<f64>],
    augmenter: &Augmenter,
    encoder: &Encoder,
    projection_head: &ProjectionHead,
    temperature: f64,
) -> f64 {
    let projections: Vec<(Array1<f64>, Array1<f64>)> = windows
        .iter()
        .map(|w| {
            let (aug_a, aug_b) = augmenter.create_positive_pair(w);
            let h_a = encoder.forward(&aug_a);
            let h_b = encoder.forward(&aug_b);
            let z_a = projection_head.forward(&h_a);
            let z_b = projection_head.forward(&h_b);
            (z_a, z_b)
        })
        .collect();

    nt_xent_loss(&projections, temperature)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_augmentation_preserves_length() {
        let config = AugmentationConfig::default();
        let augmenter = Augmenter::new(config);
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let augmented = augmenter.augment(&data);
        assert_eq!(data.len(), augmented.len());
    }

    #[test]
    fn test_encoder_output_dim() {
        let encoder = Encoder::new(10, 32, 16);
        let input = Array1::from_vec(vec![1.0; 10]);
        let output = encoder.forward(&input);
        assert_eq!(output.len(), 16);
    }

    #[test]
    fn test_projection_head_output_dim() {
        let proj = ProjectionHead::new(16, 32, 8);
        let input = Array1::from_vec(vec![1.0; 16]);
        let output = proj.forward(&input);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6, "Cosine similarity of identical vectors should be 1.0, got {}", sim);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 1.0]);
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "Cosine similarity of orthogonal vectors should be 0.0, got {}", sim);
    }

    #[test]
    fn test_nt_xent_loss_positive() {
        let proj_a = Array1::from_vec(vec![1.0, 0.5, 0.3]);
        let proj_b = Array1::from_vec(vec![0.9, 0.6, 0.2]);
        let projections = vec![(proj_a, proj_b)];
        let loss = nt_xent_loss(&projections, 0.1);
        assert!(loss.is_finite(), "NT-Xent loss should be finite");
    }

    #[test]
    fn test_linear_probe_predict() {
        let probe = LinearProbe::new(8, 3, 0.01);
        let repr = Array1::from_vec(vec![1.0; 8]);
        let class = probe.predict(&repr);
        assert!(class < 3, "Predicted class should be < num_classes");
    }

    #[test]
    fn test_zscore_normalize() {
        let data = Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        let normalized = zscore_normalize(&data);
        let mean: f64 = normalized.sum() / normalized.len() as f64;
        assert!(mean.abs() < 1e-10, "Z-scored data should have mean ~0, got {}", mean);
    }

    #[test]
    fn test_create_windows() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let windows = create_windows(&data, 5, 2);
        assert_eq!(windows.len(), 3); // [0..5], [2..7], [4..9]
        assert_eq!(windows[0].len(), 5);
    }

    #[test]
    fn test_momentum_encoder_creation() {
        let moco = MomentumEncoder::new(10, 32, 16, 8, 0.999, 100);
        let input = Array1::from_vec(vec![1.0; 10]);
        let q = moco.encode_query(&input);
        let k = moco.encode_key(&input);
        assert_eq!(q.len(), 8);
        assert_eq!(k.len(), 8);
        // Initially query and key encoders are identical, so outputs should match
        for i in 0..q.len() {
            assert!(
                (q[i] - k[i]).abs() < 1e-10,
                "Initial q and k should be equal"
            );
        }
    }

    #[test]
    fn test_simclr_step_returns_finite_loss() {
        let augmenter = Augmenter::new(AugmentationConfig::default());
        let encoder = Encoder::new(10, 32, 16);
        let proj_head = ProjectionHead::new(16, 16, 8);
        let windows = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            Array1::from_vec(vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
        ];
        let loss = simclr_step(&windows, &augmenter, &encoder, &proj_head, 0.1);
        assert!(loss.is_finite(), "SimCLR step loss should be finite, got {}", loss);
    }
}
