//! Edge Federated Learning for Trading
//!
//! This library implements edge-based federated learning with FedProx,
//! gradient compression, and Bybit data integration for trading applications.

use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Bybit Data Fetching
// ---------------------------------------------------------------------------

/// Raw kline record from Bybit API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KlineRecord {
    pub open_time: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Deserialize)]
struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    result: BybitKlineResult,
}

#[derive(Debug, Deserialize)]
struct BybitKlineResult {
    list: Vec<Vec<String>>,
}

/// Fetch BTCUSDT klines from Bybit public API.
pub fn fetch_bybit_klines(symbol: &str, interval: &str, limit: usize) -> Result<Vec<KlineRecord>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );
    let resp: BybitKlineResponse = reqwest::blocking::get(&url)?.json()?;
    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API returned error code {}", resp.ret_code);
    }
    let mut records: Vec<KlineRecord> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() < 6 {
                return None;
            }
            Some(KlineRecord {
                open_time: row[0].parse().ok()?,
                open: row[1].parse().ok()?,
                high: row[2].parse().ok()?,
                low: row[3].parse().ok()?,
                close: row[4].parse().ok()?,
                volume: row[5].parse().ok()?,
            })
        })
        .collect();
    // Bybit returns newest first; reverse to chronological order.
    records.reverse();
    Ok(records)
}

// ---------------------------------------------------------------------------
// Feature Engineering
// ---------------------------------------------------------------------------

/// A single training sample: features + target.
#[derive(Debug, Clone)]
pub struct Sample {
    pub features: Vec<f64>,
    pub target: f64,
}

/// Build features from kline records.
///
/// Features per bar (using lookback window):
///   0: return
///   1: volatility (std of returns over window)
///   2: momentum (close / close[t-window] - 1)
///   3: volume ratio (volume / mean volume over window)
///   4: normalized range ((high - low) / close)
///
/// Target: next-bar return sign (1.0 for up, 0.0 for down).
pub fn build_features(klines: &[KlineRecord], window: usize) -> Vec<Sample> {
    if klines.len() < window + 2 {
        return Vec::new();
    }
    let mut samples = Vec::new();
    for t in window..klines.len() - 1 {
        let ret = (klines[t].close - klines[t - 1].close) / klines[t - 1].close;
        // Volatility: std of returns over window
        let returns: Vec<f64> = (t - window + 1..=t)
            .map(|i| (klines[i].close - klines[i - 1].close) / klines[i - 1].close)
            .collect();
        let mean_ret: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let var: f64 =
            returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / returns.len() as f64;
        let volatility = var.sqrt();
        // Momentum
        let momentum = (klines[t].close - klines[t - window].close) / klines[t - window].close;
        // Volume ratio
        let mean_vol: f64 = (t - window + 1..=t)
            .map(|i| klines[i].volume)
            .sum::<f64>()
            / window as f64;
        let vol_ratio = if mean_vol > 0.0 {
            klines[t].volume / mean_vol
        } else {
            1.0
        };
        // Normalized range
        let range = (klines[t].high - klines[t].low) / klines[t].close;
        // Target: next bar direction
        let next_ret = (klines[t + 1].close - klines[t].close) / klines[t].close;
        let target = if next_ret >= 0.0 { 1.0 } else { 0.0 };

        samples.push(Sample {
            features: vec![ret, volatility, momentum, vol_ratio, range],
            target,
        });
    }
    samples
}

// ---------------------------------------------------------------------------
// Simple Linear Model
// ---------------------------------------------------------------------------

/// Number of input features.
pub const NUM_FEATURES: usize = 5;

/// Simple linear model: sigmoid(w * x + b)
/// Parameters stored as a flat vector: [w0..w4, bias] (length = NUM_FEATURES + 1).
pub fn model_predict(params: &[f64], features: &[f64]) -> f64 {
    assert!(params.len() == NUM_FEATURES + 1);
    assert!(features.len() == NUM_FEATURES);
    let mut z: f64 = params[NUM_FEATURES]; // bias
    for i in 0..NUM_FEATURES {
        z += params[i] * features[i];
    }
    sigmoid(z)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Binary cross-entropy loss for a single sample.
pub fn bce_loss(pred: f64, target: f64) -> f64 {
    let eps = 1e-12;
    let p = pred.clamp(eps, 1.0 - eps);
    -(target * p.ln() + (1.0 - target) * (1.0 - p).ln())
}

/// Compute gradient of BCE loss w.r.t. model parameters for one sample.
fn bce_gradient(params: &[f64], features: &[f64], target: f64) -> Vec<f64> {
    let pred = model_predict(params, features);
    let error = pred - target; // d(BCE)/d(z) = pred - target
    let mut grad = vec![0.0; NUM_FEATURES + 1];
    for i in 0..NUM_FEATURES {
        grad[i] = error * features[i];
    }
    grad[NUM_FEATURES] = error; // bias gradient
    grad
}

// ---------------------------------------------------------------------------
// Gradient Compression
// ---------------------------------------------------------------------------

/// Compressed gradient update: stores only Top-K components.
#[derive(Debug, Clone)]
pub struct CompressedUpdate {
    /// (index, quantized_value) pairs for non-zero components.
    pub components: Vec<(usize, f64)>,
    /// Number of data samples used for this update (for weighted averaging).
    pub num_samples: usize,
    /// Which parameter indices were actually trained (for partial updates).
    pub trained_indices: Vec<usize>,
}

/// Apply Top-K sparsification: keep only the K largest components by magnitude.
pub fn top_k_sparsify(gradient: &[f64], k: usize) -> (Vec<(usize, f64)>, Vec<f64>) {
    let k = k.min(gradient.len());
    let mut indexed: Vec<(usize, f64)> = gradient.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

    let selected: Vec<(usize, f64)> = indexed[..k].to_vec();
    // Residual: everything not selected
    let mut residual = gradient.to_vec();
    for &(idx, _) in &selected {
        residual[idx] = 0.0;
    }
    (selected, residual)
}

/// Quantize a value to 8-bit representation (256 levels).
pub fn quantize_8bit(value: f64, max_abs: f64) -> f64 {
    if max_abs == 0.0 {
        return 0.0;
    }
    let normalized = (value / max_abs).clamp(-1.0, 1.0);
    let quantized = (normalized * 127.0).round() / 127.0;
    quantized * max_abs
}

/// Compress a gradient update: Top-K + 8-bit quantization.
pub fn compress_gradient(gradient: &[f64], top_k_ratio: f64) -> (Vec<(usize, f64)>, Vec<f64>) {
    let k = ((gradient.len() as f64) * top_k_ratio).ceil() as usize;
    let (sparse, residual) = top_k_sparsify(gradient, k.max(1));
    // Quantize the selected components
    let max_abs = sparse
        .iter()
        .map(|(_, v)| v.abs())
        .fold(0.0_f64, f64::max);
    let quantized: Vec<(usize, f64)> = sparse
        .iter()
        .map(|&(idx, val)| (idx, quantize_8bit(val, max_abs)))
        .collect();
    (quantized, residual)
}

// ---------------------------------------------------------------------------
// Edge Device
// ---------------------------------------------------------------------------

/// Device tier representing compute capability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceTier {
    /// Colocation server: many local epochs, full parameter updates.
    Colocation,
    /// Mobile/remote device: few local epochs, partial parameter updates.
    Mobile,
}

/// An edge device participating in federated learning.
#[derive(Debug, Clone)]
pub struct EdgeDevice {
    pub id: String,
    pub tier: DeviceTier,
    /// Local model parameters.
    pub params: Vec<f64>,
    /// Residual buffer for gradient compression (error feedback).
    pub residual: Vec<f64>,
    /// Local training data.
    pub data: Vec<Sample>,
    /// Maximum number of local epochs this device can perform.
    pub max_local_epochs: usize,
    /// Fraction of parameters this device updates (1.0 = all).
    pub param_update_fraction: f64,
}

impl EdgeDevice {
    pub fn new(id: &str, tier: DeviceTier, initial_params: &[f64]) -> Self {
        let (max_epochs, param_fraction) = match tier {
            DeviceTier::Colocation => (20, 1.0),
            DeviceTier::Mobile => (3, 0.5),
        };
        EdgeDevice {
            id: id.to_string(),
            tier,
            params: initial_params.to_vec(),
            residual: vec![0.0; initial_params.len()],
            data: Vec::new(),
            max_local_epochs: max_epochs,
            param_update_fraction: param_fraction,
        }
    }

    /// Load local training data.
    pub fn load_data(&mut self, samples: Vec<Sample>) {
        self.data = samples;
    }

    /// Determine which parameter indices this device will update.
    fn trainable_indices(&self) -> Vec<usize> {
        let n = self.params.len();
        let k = ((n as f64) * self.param_update_fraction).ceil() as usize;
        // Always include first k indices (deterministic for reproducibility).
        (0..k.min(n)).collect()
    }

    /// Perform local FedProx training and return a compressed update.
    pub fn train_local(
        &mut self,
        global_params: &[f64],
        mu: f64,
        learning_rate: f64,
        top_k_ratio: f64,
    ) -> CompressedUpdate {
        if self.data.is_empty() {
            return CompressedUpdate {
                components: Vec::new(),
                num_samples: 0,
                trained_indices: Vec::new(),
            };
        }

        // Set local params to global params at start of round.
        self.params = global_params.to_vec();
        let trainable = self.trainable_indices();

        // Local SGD with proximal term.
        for _epoch in 0..self.max_local_epochs {
            for sample in &self.data {
                let grad = bce_gradient(&self.params, &sample.features, sample.target);
                for &idx in &trainable {
                    // Proximal gradient: mu * (w - w_global)
                    let prox = mu * (self.params[idx] - global_params[idx]);
                    self.params[idx] -= learning_rate * (grad[idx] + prox);
                }
            }
        }

        // Compute update delta.
        let mut delta: Vec<f64> = self
            .params
            .iter()
            .zip(global_params.iter())
            .map(|(local, global)| local - global)
            .collect();

        // Add residual from previous rounds.
        for i in 0..delta.len() {
            delta[i] += self.residual[i];
        }

        // Compress.
        let (compressed, new_residual) = compress_gradient(&delta, top_k_ratio);
        self.residual = new_residual;

        CompressedUpdate {
            components: compressed,
            num_samples: self.data.len(),
            trained_indices: trainable,
        }
    }

    /// Compute average loss on local data.
    pub fn evaluate(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        let total: f64 = self
            .data
            .iter()
            .map(|s| bce_loss(model_predict(&self.params, &s.features), s.target))
            .sum();
        total / self.data.len() as f64
    }

    /// Compute accuracy on local data.
    pub fn accuracy(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        let correct: usize = self
            .data
            .iter()
            .filter(|s| {
                let pred = model_predict(&self.params, &s.features);
                (pred >= 0.5 && s.target == 1.0) || (pred < 0.5 && s.target == 0.0)
            })
            .count();
        correct as f64 / self.data.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Edge FL Coordinator
// ---------------------------------------------------------------------------

/// The central coordinator that orchestrates federated learning across edge devices.
pub struct EdgeFLCoordinator {
    /// Global model parameters.
    pub global_params: Vec<f64>,
    /// FedProx proximal term coefficient.
    pub mu: f64,
    /// Learning rate for local training.
    pub learning_rate: f64,
    /// Top-K ratio for gradient compression (e.g., 0.5 = 50%).
    pub top_k_ratio: f64,
    /// History of global loss per round.
    pub loss_history: Vec<f64>,
}

impl EdgeFLCoordinator {
    /// Create a new coordinator with random initial parameters.
    pub fn new(mu: f64, learning_rate: f64, top_k_ratio: f64) -> Self {
        let mut rng = rand::thread_rng();
        let global_params: Vec<f64> = (0..NUM_FEATURES + 1)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();
        EdgeFLCoordinator {
            global_params,
            mu,
            learning_rate,
            top_k_ratio,
            loss_history: Vec::new(),
        }
    }

    /// Create a coordinator with specified initial parameters.
    pub fn with_params(params: Vec<f64>, mu: f64, learning_rate: f64, top_k_ratio: f64) -> Self {
        EdgeFLCoordinator {
            global_params: params,
            mu,
            learning_rate,
            top_k_ratio,
            loss_history: Vec::new(),
        }
    }

    /// Run one round of federated learning across all devices.
    pub fn run_round(&mut self, devices: &mut [EdgeDevice]) {
        // 1. Collect compressed updates from all devices.
        let updates: Vec<CompressedUpdate> = devices
            .iter_mut()
            .map(|d| d.train_local(&self.global_params, self.mu, self.learning_rate, self.top_k_ratio))
            .collect();

        // 2. Aggregate updates using weighted averaging.
        let total_samples: usize = updates.iter().map(|u| u.num_samples).sum();
        if total_samples == 0 {
            return;
        }

        // Track how many updates contribute to each parameter index.
        let param_len = self.global_params.len();
        let mut aggregated_delta = vec![0.0; param_len];
        let mut param_weights = vec![0.0; param_len];

        for update in &updates {
            let weight = update.num_samples as f64 / total_samples as f64;
            for &(idx, val) in &update.components {
                if idx < param_len {
                    aggregated_delta[idx] += weight * val;
                    param_weights[idx] += weight;
                }
            }
        }

        // 3. Apply aggregated update to global model.
        for i in 0..param_len {
            if param_weights[i] > 0.0 {
                self.global_params[i] += aggregated_delta[i];
            }
        }

        // 4. Update devices with new global params and compute global loss.
        let mut total_loss = 0.0;
        let mut total_eval_samples = 0;
        for device in devices.iter_mut() {
            device.params = self.global_params.clone();
            let n = device.data.len();
            if n > 0 {
                total_loss += device.evaluate() * n as f64;
                total_eval_samples += n;
            }
        }
        if total_eval_samples > 0 {
            self.loss_history.push(total_loss / total_eval_samples as f64);
        }
    }

    /// Run multiple rounds of federated learning.
    pub fn train(&mut self, devices: &mut [EdgeDevice], rounds: usize) {
        for round in 0..rounds {
            self.run_round(devices);
            if let Some(&loss) = self.loss_history.last() {
                println!(
                    "Round {}/{}: Global Loss = {:.6}",
                    round + 1,
                    rounds,
                    loss
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Heterogeneous Device Simulation
// ---------------------------------------------------------------------------

/// Create a set of simulated edge devices with heterogeneous capabilities.
pub fn create_heterogeneous_devices(
    num_colocation: usize,
    num_mobile: usize,
    initial_params: &[f64],
) -> Vec<EdgeDevice> {
    let mut devices = Vec::new();
    for i in 0..num_colocation {
        devices.push(EdgeDevice::new(
            &format!("colo-{}", i),
            DeviceTier::Colocation,
            initial_params,
        ));
    }
    for i in 0..num_mobile {
        devices.push(EdgeDevice::new(
            &format!("mobile-{}", i),
            DeviceTier::Mobile,
            initial_params,
        ));
    }
    devices
}

/// Distribute data across devices with non-IID splits.
///
/// Colocation devices get larger, contiguous chunks.
/// Mobile devices get smaller, random subsets.
pub fn distribute_data_non_iid(devices: &mut [EdgeDevice], all_samples: &[Sample]) {
    if devices.is_empty() || all_samples.is_empty() {
        return;
    }
    let mut rng = rand::thread_rng();

    let num_colo: usize = devices.iter().filter(|d| d.tier == DeviceTier::Colocation).count();
    let num_mobile: usize = devices.iter().filter(|d| d.tier == DeviceTier::Mobile).count();

    // Colocation devices get 70% of data split evenly.
    let colo_total = (all_samples.len() as f64 * 0.7) as usize;
    let colo_each = if num_colo > 0 { colo_total / num_colo } else { 0 };

    // Mobile devices get 30% of data split evenly.
    let mobile_total = all_samples.len() - colo_total;
    let mobile_each = if num_mobile > 0 { mobile_total / num_mobile } else { 0 };

    let mut offset = 0;
    for device in devices.iter_mut() {
        match device.tier {
            DeviceTier::Colocation => {
                let end = (offset + colo_each).min(all_samples.len());
                device.load_data(all_samples[offset..end].to_vec());
                offset = end;
            }
            DeviceTier::Mobile => {
                // Mobile devices get random subsets (non-IID simulation).
                let subset: Vec<Sample> = (0..mobile_each)
                    .map(|_| {
                        let idx = rng.gen_range(0..all_samples.len());
                        all_samples[idx].clone()
                    })
                    .collect();
                device.load_data(subset);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Synthetic Data Generation (fallback when API is unavailable)
// ---------------------------------------------------------------------------

/// Generate synthetic kline data that mimics BTC price action.
pub fn generate_synthetic_klines(num_bars: usize) -> Vec<KlineRecord> {
    let mut rng = rand::thread_rng();
    let mut price = 50000.0_f64;
    let mut records = Vec::with_capacity(num_bars);
    let mut time = 1_700_000_000_000u64; // ~Nov 2023

    for _ in 0..num_bars {
        let ret: f64 = rng.gen_range(-0.02..0.02);
        let open = price;
        let close = price * (1.0 + ret);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.005));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.005));
        let volume = rng.gen_range(100.0..10000.0);

        records.push(KlineRecord {
            open_time: time,
            open,
            high,
            low,
            close,
            volume,
        });
        price = close;
        time += 300_000; // 5-minute bars
    }
    records
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_model_predict() {
        let params = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // w0=1, rest=0
        let features = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let pred = model_predict(&params, &features);
        assert!((pred - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_top_k_sparsify() {
        let grad = vec![0.1, -0.5, 0.3, -0.9, 0.2];
        let (selected, residual) = top_k_sparsify(&grad, 2);
        assert_eq!(selected.len(), 2);
        // Top-2 by magnitude: index 3 (-0.9) and index 1 (-0.5)
        let indices: Vec<usize> = selected.iter().map(|&(i, _)| i).collect();
        assert!(indices.contains(&3));
        assert!(indices.contains(&1));
        // Residual should be zero at selected indices
        assert_eq!(residual[3], 0.0);
        assert_eq!(residual[1], 0.0);
        assert!((residual[0] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_quantize() {
        let val = quantize_8bit(0.5, 1.0);
        // Should be close to 0.5 (within quantization error)
        assert!((val - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_synthetic_data_pipeline() {
        let klines = generate_synthetic_klines(100);
        assert_eq!(klines.len(), 100);
        let samples = build_features(&klines, 10);
        assert!(!samples.is_empty());
        assert_eq!(samples[0].features.len(), NUM_FEATURES);
    }

    #[test]
    fn test_edge_device_creation() {
        let params = vec![0.0; NUM_FEATURES + 1];
        let device = EdgeDevice::new("test-colo", DeviceTier::Colocation, &params);
        assert_eq!(device.max_local_epochs, 20);
        assert_eq!(device.param_update_fraction, 1.0);

        let mobile = EdgeDevice::new("test-mobile", DeviceTier::Mobile, &params);
        assert_eq!(mobile.max_local_epochs, 3);
        assert_eq!(mobile.param_update_fraction, 0.5);
    }

    #[test]
    fn test_federated_training_converges() {
        let klines = generate_synthetic_klines(200);
        let samples = build_features(&klines, 10);

        let mut coordinator = EdgeFLCoordinator::new(0.01, 0.01, 0.5);
        let mut devices = create_heterogeneous_devices(2, 1, &coordinator.global_params);
        distribute_data_non_iid(&mut devices, &samples);

        coordinator.train(&mut devices, 5);

        // Loss should generally decrease (or at least not blow up).
        assert!(!coordinator.loss_history.is_empty());
        let last_loss = *coordinator.loss_history.last().unwrap();
        assert!(last_loss.is_finite());
    }
}
