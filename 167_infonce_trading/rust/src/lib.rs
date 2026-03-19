//! # InfoNCE Trading
//!
//! Contrastive representation learning for market state analysis using InfoNCE loss.
//! Fetches OHLCV data from Bybit and learns embeddings that cluster similar market conditions.

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single OHLCV candlestick.
#[derive(Debug, Clone)]
pub struct OhlcvCandle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Configuration for the InfoNCE training pipeline.
#[derive(Debug, Clone)]
pub struct InfoNCEConfig {
    /// Temperature parameter for the softmax (e.g. 0.1).
    pub temperature: f64,
    /// Dimensionality of the output embedding.
    pub embedding_dim: usize,
    /// Number of candles per observation window.
    pub window_size: usize,
    /// Maximum distance (in windows) for a positive pair.
    pub positive_range: usize,
    /// Number of negative samples per anchor.
    pub num_negatives: usize,
    /// Learning rate for SGD.
    pub learning_rate: f64,
    /// Number of training epochs.
    pub epochs: usize,
}

impl Default for InfoNCEConfig {
    fn default() -> Self {
        Self {
            temperature: 0.1,
            embedding_dim: 32,
            window_size: 5,
            positive_range: 3,
            num_negatives: 8,
            learning_rate: 0.001,
            epochs: 50,
        }
    }
}

/// A single contrastive training sample.
#[derive(Debug, Clone)]
pub struct ContrastiveSample {
    pub anchor: Array1<f64>,
    pub positive: Array1<f64>,
    pub negatives: Vec<Array1<f64>>,
}

// ---------------------------------------------------------------------------
// Bybit API types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i64,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub list: Vec<Vec<String>>,
}

// ---------------------------------------------------------------------------
// Similarity & loss functions
// ---------------------------------------------------------------------------

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let dot = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Temperature-scaled softmax over a slice of logits.
/// Returns a `Vec<f64>` of probabilities summing to 1.
pub fn temperature_softmax(logits: &[f64], temperature: f64) -> Vec<f64> {
    let scaled: Vec<f64> = logits.iter().map(|&l| l / temperature).collect();
    let max = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|&s| (s - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Compute the InfoNCE loss for one sample.
///
/// ```text
/// L = -log( exp(sim(q, k+) / τ)  /  Σ_i exp(sim(q, k_i) / τ) )
/// ```
///
/// The sum runs over the positive **and** all negatives.
pub fn infonce_loss(
    query: &Array1<f64>,
    positive: &Array1<f64>,
    negatives: &[Array1<f64>],
    temperature: f64,
) -> f64 {
    let pos_sim = cosine_similarity(query, positive) / temperature;

    let neg_sims: Vec<f64> = negatives
        .iter()
        .map(|n| cosine_similarity(query, n) / temperature)
        .collect();

    // Log-sum-exp trick for numerical stability
    let max_sim = neg_sims
        .iter()
        .cloned()
        .fold(pos_sim, f64::max);

    let pos_exp = (pos_sim - max_sim).exp();
    let neg_sum: f64 = neg_sims.iter().map(|&s| (s - max_sim).exp()).sum();

    -(pos_exp / (pos_exp + neg_sum)).ln()
}

/// Compute the average InfoNCE loss over a batch of contrastive samples.
pub fn batch_infonce_loss(samples: &[ContrastiveSample], encoder: &Encoder, temperature: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let total: f64 = samples
        .iter()
        .map(|s| {
            let q = encoder.forward(&s.anchor);
            let p = encoder.forward(&s.positive);
            let negs: Vec<Array1<f64>> = s.negatives.iter().map(|n| encoder.forward(n)).collect();
            infonce_loss(&q, &p, &negs, temperature)
        })
        .sum();
    total / samples.len() as f64
}

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------

/// Number of features extracted per candle.
pub const FEATURES_PER_CANDLE: usize = 6;

/// Extract a normalised feature vector from a single candle.
///
/// Features: [return, hl_range, volume_ratio, body_ratio, upper_shadow, lower_shadow]
fn candle_features(candle: &OhlcvCandle, avg_volume: f64) -> [f64; FEATURES_PER_CANDLE] {
    let open = candle.open;
    let high = candle.high;
    let low = candle.low;
    let close = candle.close;

    let ret = if open.abs() > 1e-12 { (close - open) / open } else { 0.0 };
    let hl = if open.abs() > 1e-12 { (high - low) / open } else { 0.0 };
    let vol_ratio = if avg_volume > 1e-12 { candle.volume / avg_volume } else { 1.0 };

    let range = high - low;
    let (body_ratio, upper_shadow, lower_shadow) = if range > 1e-12 {
        let body = (close - open).abs() / range;
        let upper = (high - close.max(open)) / range;
        let lower = (close.min(open) - low) / range;
        (body, upper, lower)
    } else {
        (0.0, 0.0, 0.0)
    };

    [ret, hl, vol_ratio, body_ratio, upper_shadow, lower_shadow]
}

/// Build a flat feature vector for a window of candles.
pub fn window_features(candles: &[OhlcvCandle]) -> Array1<f64> {
    let avg_volume: f64 = candles.iter().map(|c| c.volume).sum::<f64>() / candles.len() as f64;
    let mut feats = Vec::with_capacity(candles.len() * FEATURES_PER_CANDLE);
    for c in candles {
        feats.extend_from_slice(&candle_features(c, avg_volume));
    }
    Array1::from_vec(feats)
}

// ---------------------------------------------------------------------------
// Contrastive pair generation
// ---------------------------------------------------------------------------

/// Generate contrastive training samples from a sequence of candles.
pub fn generate_contrastive_pairs(
    candles: &[OhlcvCandle],
    config: &InfoNCEConfig,
) -> Vec<ContrastiveSample> {
    let mut rng = rand::thread_rng();
    let n = candles.len();
    if n < config.window_size + config.positive_range + 1 {
        return Vec::new();
    }

    let max_start = n - config.window_size;
    let mut samples = Vec::new();

    for anchor_start in 0..=max_start {
        // --- positive: a nearby window within positive_range ---
        let pos_low = anchor_start.saturating_sub(config.positive_range);
        let pos_high = (anchor_start + config.positive_range).min(max_start);
        // Avoid picking the anchor itself
        let candidates: Vec<usize> = (pos_low..=pos_high)
            .filter(|&p| p != anchor_start)
            .collect();
        if candidates.is_empty() {
            continue;
        }
        let pos_start = *candidates.choose(&mut rng).unwrap();

        // --- negatives: random windows far from anchor ---
        let far_threshold = config.positive_range * 2;
        let neg_candidates: Vec<usize> = (0..=max_start)
            .filter(|&p| {
                let dist = if p > anchor_start { p - anchor_start } else { anchor_start - p };
                dist > far_threshold
            })
            .collect();

        if neg_candidates.len() < config.num_negatives {
            continue;
        }

        let neg_starts: Vec<usize> = neg_candidates
            .choose_multiple(&mut rng, config.num_negatives)
            .cloned()
            .collect();

        let anchor_feat = window_features(&candles[anchor_start..anchor_start + config.window_size]);
        let pos_feat = window_features(&candles[pos_start..pos_start + config.window_size]);
        let neg_feats: Vec<Array1<f64>> = neg_starts
            .iter()
            .map(|&s| window_features(&candles[s..s + config.window_size]))
            .collect();

        samples.push(ContrastiveSample {
            anchor: anchor_feat,
            positive: pos_feat,
            negatives: neg_feats,
        });
    }

    samples
}

// ---------------------------------------------------------------------------
// Encoder (simple feedforward)
// ---------------------------------------------------------------------------

/// Simple feedforward encoder: input → Dense(hidden1, ReLU) → Dense(hidden2, ReLU) → Dense(embed) → L2 norm
#[derive(Debug, Clone)]
pub struct Encoder {
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
    pub w3: Array2<f64>,
    pub b3: Array1<f64>,
}

impl Encoder {
    /// Create a new encoder with Xavier-initialised weights.
    pub fn new(input_dim: usize, hidden1: usize, hidden2: usize, embed_dim: usize) -> Self {
        let mut rng = rand::thread_rng();

        let mut xavier = |fan_in: usize, fan_out: usize| -> Array2<f64> {
            let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
            Array2::from_shape_fn((fan_in, fan_out), |_| rng.gen_range(-limit..limit))
        };

        Self {
            w1: xavier(input_dim, hidden1),
            b1: Array1::zeros(hidden1),
            w2: xavier(hidden1, hidden2),
            b2: Array1::zeros(hidden2),
            w3: xavier(hidden2, embed_dim),
            b3: Array1::zeros(embed_dim),
        }
    }

    /// Forward pass — returns L2-normalised embedding.
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        // Layer 1
        let h1 = self.w1.t().dot(x) + &self.b1;
        let h1 = h1.mapv(|v| v.max(0.0)); // ReLU

        // Layer 2
        let h2 = self.w2.t().dot(&h1) + &self.b2;
        let h2 = h2.mapv(|v| v.max(0.0)); // ReLU

        // Layer 3 (projection)
        let out = self.w3.t().dot(&h2) + &self.b3;

        // L2 normalisation
        let norm = out.dot(&out).sqrt();
        if norm > 1e-12 {
            out / norm
        } else {
            out
        }
    }

    /// Train the encoder for one epoch using finite-difference gradient estimation.
    /// Returns the mean loss over the epoch.
    pub fn train_epoch(
        &mut self,
        samples: &[ContrastiveSample],
        config: &InfoNCEConfig,
    ) -> f64 {
        let eps = 1e-4;
        let lr = config.learning_rate;

        let base_loss = batch_infonce_loss(samples, self, config.temperature);

        // Update w1
        self.perturb_and_update_w(&samples, config, eps, lr, 1);
        // Update w2
        self.perturb_and_update_w(&samples, config, eps, lr, 2);
        // Update w3
        self.perturb_and_update_w(&samples, config, eps, lr, 3);

        base_loss
    }

    /// Perturb a weight matrix and perform a gradient step.
    fn perturb_and_update_w(
        &mut self,
        samples: &[ContrastiveSample],
        config: &InfoNCEConfig,
        eps: f64,
        lr: f64,
        layer: usize,
    ) {
        let (rows, cols) = match layer {
            1 => (self.w1.nrows(), self.w1.ncols()),
            2 => (self.w2.nrows(), self.w2.ncols()),
            3 => (self.w3.nrows(), self.w3.ncols()),
            _ => return,
        };

        // Sample a random subset of weight indices to keep computation manageable
        let mut rng = rand::thread_rng();
        let total_params = rows * cols;
        let sample_count = total_params.min(64); // limit to 64 parameters per update
        let indices: Vec<(usize, usize)> = (0..sample_count)
            .map(|_| (rng.gen_range(0..rows), rng.gen_range(0..cols)))
            .collect();

        let base_loss = batch_infonce_loss(samples, self, config.temperature);

        for &(i, j) in &indices {
            // Perturb
            match layer {
                1 => self.w1[[i, j]] += eps,
                2 => self.w2[[i, j]] += eps,
                3 => self.w3[[i, j]] += eps,
                _ => {}
            }

            let perturbed_loss = batch_infonce_loss(samples, self, config.temperature);
            let grad = (perturbed_loss - base_loss) / eps;

            // Restore and apply gradient
            match layer {
                1 => {
                    self.w1[[i, j]] -= eps;
                    self.w1[[i, j]] -= lr * grad;
                }
                2 => {
                    self.w2[[i, j]] -= eps;
                    self.w2[[i, j]] -= lr * grad;
                }
                3 => {
                    self.w3[[i, j]] -= eps;
                    self.w3[[i, j]] -= lr * grad;
                }
                _ => {}
            }
        }
    }

    /// Train for multiple epochs, returning the loss history.
    pub fn train(
        &mut self,
        samples: &[ContrastiveSample],
        config: &InfoNCEConfig,
    ) -> Vec<f64> {
        let mut losses = Vec::with_capacity(config.epochs);
        for epoch in 0..config.epochs {
            let loss = self.train_epoch(samples, config);
            losses.push(loss);
            if epoch % 10 == 0 || epoch == config.epochs - 1 {
                println!("Epoch {:>4} | Loss: {:.6}", epoch, loss);
            }
        }
        losses
    }
}

// ---------------------------------------------------------------------------
// Bybit API
// ---------------------------------------------------------------------------

/// Fetch OHLCV candles from Bybit's V5 public API.
///
/// * `symbol` – e.g. `"BTCUSDT"`
/// * `interval` – e.g. `"5"` for 5-minute candles
/// * `limit` – number of candles to fetch (max 200)
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<OhlcvCandle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let resp: BybitResponse = reqwest::get(&url).await?.json().await?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API returned error code {}", resp.ret_code);
    }

    let mut candles: Vec<OhlcvCandle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() < 7 {
                return None;
            }
            Some(OhlcvCandle {
                timestamp: row[0].parse().ok()?,
                open: row[1].parse().ok()?,
                high: row[2].parse().ok()?,
                low: row[3].parse().ok()?,
                close: row[4].parse().ok()?,
                volume: row[5].parse().ok()?,
            })
        })
        .collect();

    // Bybit returns newest-first; reverse to chronological order
    candles.reverse();

    Ok(candles)
}

/// Blocking version of [`fetch_bybit_klines`] for non-async contexts.
pub fn fetch_bybit_klines_blocking(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<OhlcvCandle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let resp: BybitResponse = reqwest::blocking::get(&url)?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API returned error code {}", resp.ret_code);
    }

    let mut candles: Vec<OhlcvCandle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() < 7 {
                return None;
            }
            Some(OhlcvCandle {
                timestamp: row[0].parse().ok()?,
                open: row[1].parse().ok()?,
                high: row[2].parse().ok()?,
                low: row[3].parse().ok()?,
                close: row[4].parse().ok()?,
                volume: row[5].parse().ok()?,
            })
        })
        .collect();

    candles.reverse();

    Ok(candles)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 1.0]);
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-9);
    }

    #[test]
    fn test_temperature_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = temperature_softmax(&logits, 0.1);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_infonce_loss_positive() {
        let q = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let p = Array1::from_vec(vec![0.9, 0.1, 0.0]);
        let n1 = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        let n2 = Array1::from_vec(vec![0.0, 0.0, 1.0]);
        let loss = infonce_loss(&q, &p, &[n1, n2], 0.1);
        assert!(loss >= 0.0);
        // With a strong positive and weak negatives, loss should be small
        assert!(loss < 1.0);
    }

    #[test]
    fn test_encoder_output_normalised() {
        let enc = Encoder::new(30, 128, 64, 32);
        let input = Array1::from_vec(vec![0.5; 30]);
        let out = enc.forward(&input);
        let norm = out.dot(&out).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_window_features_length() {
        let candles: Vec<OhlcvCandle> = (0..5)
            .map(|i| OhlcvCandle {
                timestamp: i as u64,
                open: 100.0,
                high: 105.0,
                low: 95.0,
                close: 102.0,
                volume: 1000.0,
            })
            .collect();
        let feats = window_features(&candles);
        assert_eq!(feats.len(), 5 * FEATURES_PER_CANDLE);
    }
}
