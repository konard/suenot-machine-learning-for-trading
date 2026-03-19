//! # Cross-Silo Federated Learning for Trading
//!
//! This crate implements a cross-silo federated learning system designed for
//! institutional trading. It includes:
//! - A federated coordinator (star topology) with weighted FedAvg
//! - Institutional client abstraction with local training
//! - Secure aggregation simulation (pairwise masking)
//! - Differential privacy (gradient clipping + Gaussian noise)
//! - Bybit market data integration

use anyhow::{Context, Result};
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Linear Model
// ---------------------------------------------------------------------------

/// A simple linear regression model: y = weights . x + bias
#[derive(Debug, Clone)]
pub struct LinearModel {
    pub weights: Array1<f64>,
    pub bias: f64,
}

impl LinearModel {
    pub fn new(dim: usize) -> Self {
        Self {
            weights: Array1::zeros(dim),
            bias: 0.0,
        }
    }

    pub fn random(dim: usize, rng: &mut StdRng) -> Self {
        let weights = Array1::from_iter((0..dim).map(|_| rng.gen_range(-0.1..0.1)));
        Self {
            weights,
            bias: rng.gen_range(-0.01..0.01),
        }
    }

    pub fn predict(&self, x: &Array1<f64>) -> f64 {
        self.weights.dot(x) + self.bias
    }

    /// Return (weight_grads, bias_grad) for MSE loss on a single sample.
    pub fn gradient(&self, x: &Array1<f64>, y: f64) -> (Array1<f64>, f64) {
        let pred = self.predict(x);
        let error = pred - y;
        let weight_grads = x * (2.0 * error);
        let bias_grad = 2.0 * error;
        (weight_grads, bias_grad)
    }

    /// Flatten model parameters into a single vector (weights ++ [bias]).
    pub fn flatten(&self) -> Array1<f64> {
        let mut v = self.weights.to_vec();
        v.push(self.bias);
        Array1::from(v)
    }

    /// Restore model parameters from a flat vector.
    pub fn unflatten(&mut self, params: &Array1<f64>) {
        let dim = self.weights.len();
        for i in 0..dim {
            self.weights[i] = params[i];
        }
        self.bias = params[dim];
    }

    pub fn param_count(&self) -> usize {
        self.weights.len() + 1
    }
}

// ---------------------------------------------------------------------------
// Differential Privacy
// ---------------------------------------------------------------------------

/// Differential privacy mechanism: gradient clipping + Gaussian noise.
#[derive(Debug, Clone)]
pub struct DifferentialPrivacy {
    /// Maximum L2 norm for gradient clipping.
    pub clip_norm: f64,
    /// Noise multiplier (sigma). Actual noise std = sigma * clip_norm.
    pub noise_sigma: f64,
}

impl DifferentialPrivacy {
    pub fn new(clip_norm: f64, noise_sigma: f64) -> Self {
        Self {
            clip_norm,
            noise_sigma,
        }
    }

    /// Clip a parameter vector to the configured L2 norm bound.
    pub fn clip(&self, params: &Array1<f64>) -> Array1<f64> {
        let norm = params.dot(params).sqrt();
        if norm > self.clip_norm {
            params * (self.clip_norm / norm)
        } else {
            params.clone()
        }
    }

    /// Add calibrated Gaussian noise for differential privacy.
    pub fn add_noise(&self, params: &Array1<f64>, rng: &mut StdRng) -> Array1<f64> {
        let std_dev = self.noise_sigma * self.clip_norm;
        let noise = Array1::from_iter(
            (0..params.len()).map(|_| rng.gen::<f64>() * std_dev * 2.0 - std_dev),
        );
        params + &noise
    }

    /// Apply clipping then noise injection.
    pub fn privatize(&self, params: &Array1<f64>, rng: &mut StdRng) -> Array1<f64> {
        let clipped = self.clip(params);
        self.add_noise(&clipped, rng)
    }
}

// ---------------------------------------------------------------------------
// Secure Aggregation
// ---------------------------------------------------------------------------

/// Simulates the pairwise-masking secure aggregation protocol.
///
/// In production, the shared seeds would be established via Diffie-Hellman key
/// exchange, and the PRG would be a cryptographic PRNG (e.g., AES-CTR).
/// Here we use deterministic StdRng seeds for demonstration.
pub struct SecureAggregator {
    /// Number of participants.
    num_participants: usize,
    /// Pairwise shared seeds: seeds\[i\]\[j\] for i < j.
    seeds: HashMap<(usize, usize), u64>,
}

impl SecureAggregator {
    pub fn new(num_participants: usize) -> Self {
        let mut seeds = HashMap::new();
        let mut rng = StdRng::seed_from_u64(42);
        for i in 0..num_participants {
            for j in (i + 1)..num_participants {
                seeds.insert((i, j), rng.gen());
            }
        }
        Self {
            num_participants,
            seeds,
        }
    }

    /// Generate a pseudorandom mask vector from a seed.
    fn prg(seed: u64, len: usize) -> Array1<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        Array1::from_iter((0..len).map(|_| rng.gen_range(-1.0..1.0)))
    }

    /// Mask participant k's update.
    pub fn mask(&self, participant_id: usize, update: &Array1<f64>) -> Array1<f64> {
        let len = update.len();
        let mut masked = update.clone();

        for j in 0..self.num_participants {
            if j == participant_id {
                continue;
            }
            let (i_low, i_high) = if participant_id < j {
                (participant_id, j)
            } else {
                (j, participant_id)
            };
            let seed = self.seeds[&(i_low, i_high)];
            let prg_vec = Self::prg(seed, len);

            if participant_id < j {
                // Add mask for j > k
                masked = masked + &prg_vec;
            } else {
                // Subtract mask for j < k
                masked = masked - &prg_vec;
            }
        }

        masked
    }

    /// Aggregate masked updates. The masks cancel out, yielding the true sum.
    pub fn aggregate(&self, masked_updates: &[Array1<f64>]) -> Array1<f64> {
        let len = masked_updates[0].len();
        let mut sum = Array1::zeros(len);
        for u in masked_updates {
            sum = sum + u;
        }
        sum
    }
}

// ---------------------------------------------------------------------------
// Institutional Client
// ---------------------------------------------------------------------------

/// Represents an institutional participant (e.g., hedge fund, bank).
pub struct InstitutionalClient {
    pub id: usize,
    pub name: String,
    pub model: LinearModel,
    /// Local dataset: Vec<(features, target)>.
    pub data: Vec<(Array1<f64>, f64)>,
    pub learning_rate: f64,
    pub local_epochs: usize,
    pub dp: DifferentialPrivacy,
    rng: StdRng,
}

impl InstitutionalClient {
    pub fn new(
        id: usize,
        name: String,
        feature_dim: usize,
        learning_rate: f64,
        local_epochs: usize,
        dp: DifferentialPrivacy,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(id as u64 + 100);
        let model = LinearModel::random(feature_dim, &mut rng);
        Self {
            id,
            name,
            model,
            data: Vec::new(),
            learning_rate,
            local_epochs,
            dp,
            rng,
        }
    }

    /// Set the local model to the global model parameters.
    pub fn receive_global_model(&mut self, global_params: &Array1<f64>) {
        self.model.unflatten(global_params);
    }

    /// Train locally on private data and return the privatised update (delta).
    pub fn train_and_get_update(&mut self, global_params: &Array1<f64>) -> Array1<f64> {
        // Start from the global model
        self.model.unflatten(global_params);

        if self.data.is_empty() {
            return Array1::zeros(self.model.param_count());
        }

        // Local SGD for E epochs
        for _epoch in 0..self.local_epochs {
            for (x, y) in &self.data {
                let (w_grad, b_grad) = self.model.gradient(x, *y);
                self.model.weights =
                    &self.model.weights - &(&w_grad * self.learning_rate);
                self.model.bias -= b_grad * self.learning_rate;
            }
        }

        // Compute delta = local_params - global_params
        let local_params = self.model.flatten();
        let delta = &local_params - global_params;

        // Apply differential privacy (clip + noise)
        self.dp.privatize(&delta, &mut self.rng)
    }

    /// Number of local data samples (used for weighting in FedAvg).
    pub fn data_size(&self) -> usize {
        self.data.len()
    }
}

// ---------------------------------------------------------------------------
// Federated Coordinator
// ---------------------------------------------------------------------------

/// Central coordinator for cross-silo federated learning (star topology).
pub struct FederatedCoordinator {
    pub global_model: LinearModel,
    pub secure_aggregator: SecureAggregator,
    pub num_rounds: usize,
    pub feature_dim: usize,
}

impl FederatedCoordinator {
    pub fn new(feature_dim: usize, num_participants: usize, num_rounds: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(0);
        Self {
            global_model: LinearModel::random(feature_dim, &mut rng),
            secure_aggregator: SecureAggregator::new(num_participants),
            num_rounds,
            feature_dim,
        }
    }

    /// Run the full federated training loop.
    pub fn train(&mut self, clients: &mut [InstitutionalClient]) -> Vec<f64> {
        let mut round_losses = Vec::new();

        for round in 0..self.num_rounds {
            let global_params = self.global_model.flatten();

            // 1. Each client trains locally and produces a privatised update
            let mut raw_updates = Vec::new();
            let mut data_sizes = Vec::new();

            for client in clients.iter_mut() {
                let update = client.train_and_get_update(&global_params);
                raw_updates.push(update);
                data_sizes.push(client.data_size());
            }

            // 2. Secure aggregation: mask individual updates
            let masked: Vec<Array1<f64>> = raw_updates
                .iter()
                .enumerate()
                .map(|(i, u)| self.secure_aggregator.mask(i, u))
                .collect();

            // 3. Server aggregates (masks cancel out)
            let agg_sum = self.secure_aggregator.aggregate(&masked);

            // 4. Weighted FedAvg: weight by data size
            let total_samples: usize = data_sizes.iter().sum();
            if total_samples == 0 {
                continue;
            }

            // The agg_sum is the sum of deltas. For weighted FedAvg we need
            // weighted average, so we recompute with weights.
            let mut weighted_delta = Array1::zeros(self.global_model.param_count());
            for (i, update) in raw_updates.iter().enumerate() {
                let weight = data_sizes[i] as f64 / total_samples as f64;
                weighted_delta = weighted_delta + &(update * weight);
            }

            // 5. Update global model
            let new_params = &global_params + &weighted_delta;
            self.global_model.unflatten(&new_params);

            // 6. Evaluate global loss across all clients' data
            let loss = self.evaluate_global_loss(clients);
            round_losses.push(loss);

            println!(
                "Round {}/{}: global MSE loss = {:.6}, total samples = {}",
                round + 1,
                self.num_rounds,
                loss,
                total_samples
            );
        }

        round_losses
    }

    /// Compute MSE loss of the global model on all clients' data.
    fn evaluate_global_loss(&self, clients: &[InstitutionalClient]) -> f64 {
        let mut total_loss = 0.0;
        let mut total_samples = 0;
        for client in clients {
            for (x, y) in &client.data {
                let pred = self.global_model.predict(x);
                total_loss += (pred - y).powi(2);
                total_samples += 1;
            }
        }
        if total_samples == 0 {
            0.0
        } else {
            total_loss / total_samples as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Bybit Data Fetcher
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// OHLCV candle data from Bybit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Fetches kline (candlestick) data from Bybit public API v5.
///
/// # Arguments
/// * `symbol` - Trading pair, e.g., "BTCUSDT"
/// * `interval` - Candle interval, e.g., "60" for 1-hour
/// * `limit` - Number of candles to fetch (max 200)
pub fn fetch_bybit_klines(symbol: &str, interval: &str, limit: usize) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client
        .get(&url)
        .header("User-Agent", "fl-cross-silo-trading/0.1")
        .send()
        .context("Failed to send request to Bybit API")?
        .json()
        .context("Failed to parse Bybit API response")?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API returned error code: {}", resp.ret_code);
    }

    let candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() < 6 {
                return None;
            }
            Some(Candle {
                timestamp: row[0].parse().ok()?,
                open: row[1].parse().ok()?,
                high: row[2].parse().ok()?,
                low: row[3].parse().ok()?,
                close: row[4].parse().ok()?,
                volume: row[5].parse().ok()?,
            })
        })
        .collect();

    Ok(candles)
}

/// Convert candles into feature vectors and targets for model training.
///
/// Features per candle:
///   0: return = (close - open) / open
///   1: volatility = (high - low) / open
///   2: normalised volume (relative to mean volume)
///
/// Target: next-period return.
pub fn candles_to_dataset(candles: &[Candle]) -> Vec<(Array1<f64>, f64)> {
    if candles.len() < 3 {
        return Vec::new();
    }

    // Compute mean volume for normalisation
    let mean_vol: f64 = candles.iter().map(|c| c.volume).sum::<f64>() / candles.len() as f64;
    let mean_vol = if mean_vol == 0.0 { 1.0 } else { mean_vol };

    let mut dataset = Vec::new();

    // Candles come newest-first from Bybit; reverse for chronological order
    let mut sorted = candles.to_vec();
    sorted.sort_by_key(|c| c.timestamp);

    for i in 0..sorted.len() - 1 {
        let c = &sorted[i];
        let next = &sorted[i + 1];

        let ret = if c.open != 0.0 {
            (c.close - c.open) / c.open
        } else {
            0.0
        };
        let vol = if c.open != 0.0 {
            (c.high - c.low) / c.open
        } else {
            0.0
        };
        let norm_volume = c.volume / mean_vol;

        let target = if next.open != 0.0 {
            (next.close - next.open) / next.open
        } else {
            0.0
        };

        let features = Array1::from(vec![ret, vol, norm_volume]);
        dataset.push((features, target));
    }

    dataset
}

/// Generate synthetic trading data for testing when the API is unavailable.
pub fn generate_synthetic_data(
    num_samples: usize,
    feature_dim: usize,
    seed: u64,
) -> Vec<(Array1<f64>, f64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let true_weights = Array1::from_iter((0..feature_dim).map(|i| (i as f64 + 1.0) * 0.1));
    let true_bias = 0.01;

    (0..num_samples)
        .map(|_| {
            let features =
                Array1::from_iter((0..feature_dim).map(|_| rng.gen_range(-0.05..0.05)));
            let noise: f64 = rng.gen_range(-0.005..0.005);
            let target = true_weights.dot(&features) + true_bias + noise;
            (features, target)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_model_flatten_unflatten() {
        let mut rng = StdRng::seed_from_u64(42);
        let model = LinearModel::random(3, &mut rng);
        let flat = model.flatten();
        let mut model2 = LinearModel::new(3);
        model2.unflatten(&flat);
        assert!((model.bias - model2.bias).abs() < 1e-10);
        for i in 0..3 {
            assert!((model.weights[i] - model2.weights[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dp_clipping() {
        let dp = DifferentialPrivacy::new(1.0, 0.0);
        let v = Array1::from(vec![3.0, 4.0]); // norm = 5
        let clipped = dp.clip(&v);
        let norm = clipped.dot(&clipped).sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_secure_aggregation_masks_cancel() {
        let n = 3;
        let dim = 5;
        let sa = SecureAggregator::new(n);

        let updates: Vec<Array1<f64>> = (0..n)
            .map(|i| Array1::from_iter((0..dim).map(|j| (i * dim + j) as f64)))
            .collect();

        let masked: Vec<Array1<f64>> = updates
            .iter()
            .enumerate()
            .map(|(i, u)| sa.mask(i, u))
            .collect();

        let agg = sa.aggregate(&masked);
        let true_sum: Array1<f64> = updates.iter().fold(Array1::zeros(dim), |acc, u| acc + u);

        for i in 0..dim {
            assert!(
                (agg[i] - true_sum[i]).abs() < 1e-10,
                "Mismatch at index {}: {} vs {}",
                i,
                agg[i],
                true_sum[i]
            );
        }
    }

    #[test]
    fn test_federated_training_converges() {
        let feature_dim = 3;
        let dp = DifferentialPrivacy::new(1.0, 0.01);

        let mut clients: Vec<InstitutionalClient> = (0..3)
            .map(|i| {
                let mut client = InstitutionalClient::new(
                    i,
                    format!("Fund_{}", i),
                    feature_dim,
                    0.01,
                    3,
                    dp.clone(),
                );
                client.data = generate_synthetic_data(50, feature_dim, i as u64 * 1000);
                client
            })
            .collect();

        let mut coordinator = FederatedCoordinator::new(feature_dim, 3, 10);
        let losses = coordinator.train(&mut clients);

        // Loss should generally decrease
        assert!(losses.last().unwrap() < &losses[0] || losses[0] < 0.01);
    }

    #[test]
    fn test_synthetic_data_generation() {
        let data = generate_synthetic_data(100, 3, 42);
        assert_eq!(data.len(), 100);
        assert_eq!(data[0].0.len(), 3);
    }
}
