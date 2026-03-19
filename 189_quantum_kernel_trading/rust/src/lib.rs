//! Quantum Kernel Trading Library
//!
//! Implements quantum kernel methods for market regime detection:
//! - Quantum state vector simulation (2^n complex amplitudes)
//! - Quantum feature map (angle encoding of financial features)
//! - Fidelity kernel computation K(x,y) = |<phi(x)|phi(y)>|^2
//! - Kernel matrix construction for datasets
//! - Kernel-based classifier (kernel perceptron)
//! - Market regime labeling from volatility
//! - Bybit API data fetching

use anyhow::{Context, Result};
use ndarray::Array2;
use serde::Deserialize;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Complex number
// ---------------------------------------------------------------------------

/// Minimal complex number type for quantum state amplitudes.
#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    /// |z|^2
    pub fn norm_squared(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Complex conjugate
    pub fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Multiply two complex numbers
    pub fn mul(&self, other: &Complex) -> Complex {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    /// Add two complex numbers
    pub fn add(&self, other: &Complex) -> Complex {
        Complex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

// ---------------------------------------------------------------------------
// Quantum state
// ---------------------------------------------------------------------------

/// A quantum state represented as a vector of 2^n complex amplitudes.
#[derive(Debug, Clone)]
pub struct QuantumState {
    pub amplitudes: Vec<Complex>,
    pub num_qubits: usize,
}

impl QuantumState {
    /// Create the |0...0> state for n qubits.
    pub fn zero_state(num_qubits: usize) -> Self {
        let dim = 1 << num_qubits;
        let mut amplitudes = vec![Complex::zero(); dim];
        amplitudes[0] = Complex::new(1.0, 0.0);
        Self {
            amplitudes,
            num_qubits,
        }
    }

    /// Dimension of the Hilbert space (2^n).
    pub fn dim(&self) -> usize {
        self.amplitudes.len()
    }

    /// Compute the squared norm (should be 1 for valid states).
    pub fn norm_squared(&self) -> f64 {
        self.amplitudes.iter().map(|a| a.norm_squared()).sum()
    }
}

/// Compute the complex inner product <a|b> = sum_i conj(a_i) * b_i.
pub fn inner_product(a: &[Complex], b: &[Complex]) -> Complex {
    assert_eq!(a.len(), b.len(), "State vectors must have the same dimension");
    let mut result = Complex::zero();
    for i in 0..a.len() {
        result = result.add(&a[i].conj().mul(&b[i]));
    }
    result
}

/// Compute the fidelity kernel K(x,y) = |<phi(x)|phi(y)>|^2.
pub fn fidelity_kernel(state1: &QuantumState, state2: &QuantumState) -> f64 {
    let ip = inner_product(&state1.amplitudes, &state2.amplitudes);
    ip.norm_squared()
}

// ---------------------------------------------------------------------------
// Quantum feature map (angle encoding)
// ---------------------------------------------------------------------------

/// Quantum feature map using angle encoding.
///
/// Each feature x_i is encoded as a Y-rotation R_Y(x_i) on qubit i.
/// The resulting state is the tensor product of single-qubit states:
///   |phi(x)> = tensor_i [cos(x_i/2)|0> + sin(x_i/2)|1>]
///
/// Features are scaled to [0, pi] before encoding.
pub struct QuantumFeatureMap {
    pub num_qubits: usize,
}

impl QuantumFeatureMap {
    pub fn new(num_qubits: usize) -> Self {
        Self { num_qubits }
    }

    /// Encode a feature vector into a quantum state via angle encoding.
    ///
    /// `features` should have length == num_qubits. Each value is expected
    /// to already be in [0, pi]. If the vector is shorter, remaining qubits
    /// stay in |0>; if longer, extra features are ignored.
    pub fn encode(&self, features: &[f64]) -> QuantumState {
        let n = self.num_qubits;
        let dim = 1usize << n;
        let mut amplitudes = vec![Complex::zero(); dim];

        // Build single-qubit states
        let mut single_qubit_states: Vec<[Complex; 2]> = Vec::with_capacity(n);
        for q in 0..n {
            let angle = if q < features.len() {
                features[q]
            } else {
                0.0
            };
            let half = angle / 2.0;
            single_qubit_states.push([
                Complex::new(half.cos(), 0.0), // cos(x/2)|0>
                Complex::new(half.sin(), 0.0), // sin(x/2)|1>
            ]);
        }

        // Tensor product: iterate over all basis states
        for idx in 0..dim {
            let mut amp = Complex::new(1.0, 0.0);
            for q in 0..n {
                let bit = (idx >> (n - 1 - q)) & 1;
                amp = amp.mul(&single_qubit_states[q][bit]);
            }
            amplitudes[idx] = amp;
        }

        QuantumState {
            amplitudes,
            num_qubits: n,
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel matrix construction
// ---------------------------------------------------------------------------

/// Build the N x N kernel matrix for a dataset using the fidelity kernel.
///
/// `dataset` is a slice of feature vectors (each already scaled to [0, pi]).
pub fn build_kernel_matrix(feature_map: &QuantumFeatureMap, dataset: &[Vec<f64>]) -> Array2<f64> {
    let n = dataset.len();
    let mut matrix = Array2::<f64>::zeros((n, n));

    // Encode all states first
    let states: Vec<QuantumState> = dataset.iter().map(|x| feature_map.encode(x)).collect();

    for i in 0..n {
        matrix[[i, i]] = 1.0; // K(x,x) = 1
        for j in (i + 1)..n {
            let k = fidelity_kernel(&states[i], &states[j]);
            matrix[[i, j]] = k;
            matrix[[j, i]] = k;
        }
    }

    matrix
}

/// Compute kernel values between a single point and all training points.
pub fn compute_kernel_row(
    feature_map: &QuantumFeatureMap,
    training_states: &[QuantumState],
    new_point: &[f64],
) -> Vec<f64> {
    let new_state = feature_map.encode(new_point);
    training_states
        .iter()
        .map(|s| fidelity_kernel(s, &new_state))
        .collect()
}

// ---------------------------------------------------------------------------
// Kernel Perceptron classifier
// ---------------------------------------------------------------------------

/// A kernel perceptron classifier that operates on a precomputed kernel matrix.
///
/// Decision function: f(x) = sign( sum_i alpha_i * y_i * K(x_i, x) )
pub struct KernelPerceptron {
    pub alphas: Vec<f64>,
    pub bias: f64,
}

impl KernelPerceptron {
    pub fn new(n_samples: usize) -> Self {
        Self {
            alphas: vec![0.0; n_samples],
            bias: 0.0,
        }
    }

    /// Train the kernel perceptron on a precomputed kernel matrix.
    ///
    /// `labels` should contain +1.0 or -1.0 values.
    pub fn train(&mut self, kernel_matrix: &Array2<f64>, labels: &[f64], epochs: usize) {
        let n = labels.len();
        assert_eq!(kernel_matrix.nrows(), n);
        assert_eq!(kernel_matrix.ncols(), n);

        for _epoch in 0..epochs {
            let mut errors = 0;
            for i in 0..n {
                let prediction = self.predict_index(kernel_matrix, labels, i);
                if prediction != labels[i] {
                    self.alphas[i] += 1.0;
                    self.bias += labels[i];
                    errors += 1;
                }
            }
            if errors == 0 {
                break;
            }
        }
    }

    /// Predict for a training point (using kernel matrix row).
    fn predict_index(&self, kernel_matrix: &Array2<f64>, labels: &[f64], idx: usize) -> f64 {
        let n = labels.len();
        let mut score = self.bias;
        for j in 0..n {
            score += self.alphas[j] * labels[j] * kernel_matrix[[j, idx]];
        }
        if score >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    /// Predict for a new point given its kernel values to all training points.
    pub fn predict(&self, kernel_row: &[f64], labels: &[f64]) -> f64 {
        let mut score = self.bias;
        for j in 0..labels.len() {
            score += self.alphas[j] * labels[j] * kernel_row[j];
        }
        if score >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    /// Compute classification accuracy on the training set.
    pub fn training_accuracy(&self, kernel_matrix: &Array2<f64>, labels: &[f64]) -> f64 {
        let n = labels.len();
        let correct = (0..n)
            .filter(|&i| self.predict_index(kernel_matrix, labels, i) == labels[i])
            .count();
        correct as f64 / n as f64
    }
}

// ---------------------------------------------------------------------------
// Market regime labeling
// ---------------------------------------------------------------------------

/// Compute rolling standard deviation with the given window size.
pub fn rolling_std(data: &[f64], window: usize) -> Vec<f64> {
    if data.len() < window {
        return vec![];
    }
    let mut result = Vec::with_capacity(data.len() - window + 1);
    for i in 0..=(data.len() - window) {
        let slice = &data[i..i + window];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let var: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
        result.push(var.sqrt());
    }
    result
}

/// Compute rolling mean with the given window size.
pub fn rolling_mean(data: &[f64], window: usize) -> Vec<f64> {
    if data.len() < window {
        return vec![];
    }
    let mut result = Vec::with_capacity(data.len() - window + 1);
    for i in 0..=(data.len() - window) {
        let slice = &data[i..i + window];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        result.push(mean);
    }
    result
}

/// Assign regime labels based on rolling volatility.
///
/// +1.0 = high volatility regime (above median)
/// -1.0 = low volatility regime (below median)
pub fn label_regimes(volatilities: &[f64]) -> Vec<f64> {
    if volatilities.is_empty() {
        return vec![];
    }
    let mut sorted = volatilities.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    volatilities
        .iter()
        .map(|&v| if v >= median { 1.0 } else { -1.0 })
        .collect()
}

// ---------------------------------------------------------------------------
// Feature engineering
// ---------------------------------------------------------------------------

/// Scale a value from [min, max] to [0, pi].
pub fn scale_to_pi(value: f64, min_val: f64, max_val: f64) -> f64 {
    if (max_val - min_val).abs() < 1e-12 {
        return PI / 2.0;
    }
    let normalized = (value - min_val) / (max_val - min_val);
    normalized.clamp(0.0, 1.0) * PI
}

/// Normalize a feature column to [0, pi].
pub fn normalize_feature(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }
    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    values
        .iter()
        .map(|&v| scale_to_pi(v, min_val, max_val))
        .collect()
}

/// Engineer features from OHLCV data and return (feature_matrix, labels).
///
/// Features per sample: [log_return, volatility, relative_volume, momentum]
/// Labels: regime based on volatility (high = +1, low = -1)
///
/// `window` is the rolling window size for volatility and volume averaging.
pub fn engineer_features(candles: &[Candle], window: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = candles.len();
    if n < window + 1 {
        return (vec![], vec![]);
    }

    // Log returns
    let log_returns: Vec<f64> = (1..n)
        .map(|i| (candles[i].close / candles[i - 1].close).ln())
        .collect();

    // Rolling volatility of returns
    let vol = rolling_std(&log_returns, window);

    // Volumes
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();
    let vol_mean = rolling_mean(&volumes[1..], window); // align with returns

    // Momentum: close[i] / close[i - window] - 1
    // We need at least window+1 candles for this; align with vol output
    let start = window; // first index in log_returns that has a full vol window

    let num_samples = vol.len().min(vol_mean.len());
    if num_samples == 0 {
        return (vec![], vec![]);
    }

    // Collect raw features
    let mut raw_returns = Vec::with_capacity(num_samples);
    let mut raw_vol = Vec::with_capacity(num_samples);
    let mut raw_rel_vol = Vec::with_capacity(num_samples);
    let mut raw_momentum = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let candle_idx = start + i; // index into candles (offset by 1 for returns)
        raw_returns.push(log_returns[start - 1 + i]);
        raw_vol.push(vol[i]);

        let rel_vol = if vol_mean[i].abs() > 1e-12 {
            volumes[candle_idx] / vol_mean[i]
        } else {
            1.0
        };
        raw_rel_vol.push(rel_vol);

        let mom = if candle_idx >= window {
            candles[candle_idx].close / candles[candle_idx - window].close - 1.0
        } else {
            0.0
        };
        raw_momentum.push(mom);
    }

    // Normalize each feature to [0, pi]
    let norm_returns = normalize_feature(&raw_returns);
    let norm_vol = normalize_feature(&raw_vol);
    let norm_rel_vol = normalize_feature(&raw_rel_vol);
    let norm_momentum = normalize_feature(&raw_momentum);

    // Build feature matrix
    let features: Vec<Vec<f64>> = (0..num_samples)
        .map(|i| {
            vec![
                norm_returns[i],
                norm_vol[i],
                norm_rel_vol[i],
                norm_momentum[i],
            ]
        })
        .collect();

    // Labels from volatility
    let labels = label_regimes(&raw_vol);

    (features, labels)
}

// ---------------------------------------------------------------------------
// Bybit API data fetching
// ---------------------------------------------------------------------------

/// A single OHLCV candle.
#[derive(Debug, Clone)]
pub struct Candle {
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

/// Fetch OHLCV candles from Bybit V5 API (blocking).
///
/// Returns candles sorted chronologically (oldest first).
///
/// # Arguments
/// * `symbol` - e.g. "BTCUSDT"
/// * `interval` - e.g. "60" for 1h
/// * `limit` - number of candles (max 200)
pub fn fetch_bybit_candles(symbol: &str, interval: &str, limit: usize) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client
        .get(&url)
        .header("User-Agent", "quantum-kernel-trading/0.1")
        .send()
        .context("Failed to send request to Bybit API")?
        .json()
        .context("Failed to parse Bybit API response")?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {} (code {})", resp.ret_msg, resp.ret_code);
    }

    let mut candles: Vec<Candle> = resp
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

    // Bybit returns newest first; reverse to chronological order
    candles.reverse();

    Ok(candles)
}

/// Compute kernel alignment KA(K, K*) between the kernel matrix and ideal kernel.
///
/// Ideal kernel: K*_ij = y_i * y_j
pub fn kernel_alignment(kernel_matrix: &Array2<f64>, labels: &[f64]) -> f64 {
    let n = labels.len();
    assert_eq!(kernel_matrix.nrows(), n);

    let mut k_k_star = 0.0; // <K, K*>_F
    let mut k_norm = 0.0; // ||K||_F^2
    let mut k_star_norm = 0.0; // ||K*||_F^2

    for i in 0..n {
        for j in 0..n {
            let k_ij = kernel_matrix[[i, j]];
            let k_star_ij = labels[i] * labels[j];
            k_k_star += k_ij * k_star_ij;
            k_norm += k_ij * k_ij;
            k_star_norm += k_star_ij * k_star_ij;
        }
    }

    if k_norm < 1e-12 || k_star_norm < 1e-12 {
        return 0.0;
    }

    k_k_star / (k_norm.sqrt() * k_star_norm.sqrt())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_operations() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let prod = a.mul(&b);
        // (1+2i)(3+4i) = 3+4i+6i+8i^2 = -5+10i
        assert!((prod.re - (-5.0)).abs() < 1e-10);
        assert!((prod.im - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_state() {
        let state = QuantumState::zero_state(3);
        assert_eq!(state.dim(), 8);
        assert!((state.amplitudes[0].re - 1.0).abs() < 1e-10);
        for i in 1..8 {
            assert!(state.amplitudes[i].norm_squared() < 1e-10);
        }
    }

    #[test]
    fn test_fidelity_kernel_same_state() {
        let fm = QuantumFeatureMap::new(3);
        let features = vec![0.5, 1.0, 1.5];
        let state = fm.encode(&features);
        let k = fidelity_kernel(&state, &state);
        assert!((k - 1.0).abs() < 1e-10, "K(x,x) should be 1, got {}", k);
    }

    #[test]
    fn test_fidelity_kernel_orthogonal() {
        let fm = QuantumFeatureMap::new(1);
        let s0 = fm.encode(&[0.0]); // |0>
        let s1 = fm.encode(&[PI]); // |1>
        let k = fidelity_kernel(&s0, &s1);
        assert!(k < 1e-10, "K(|0>, |1>) should be ~0, got {}", k);
    }

    #[test]
    fn test_state_normalization() {
        let fm = QuantumFeatureMap::new(4);
        let features = vec![0.3, 1.2, 2.5, 0.8];
        let state = fm.encode(&features);
        let norm = state.norm_squared();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "State should be normalized, got {}",
            norm
        );
    }

    #[test]
    fn test_kernel_matrix_symmetric() {
        let fm = QuantumFeatureMap::new(2);
        let data = vec![vec![0.5, 1.0], vec![1.0, 0.5], vec![0.2, 2.0]];
        let km = build_kernel_matrix(&fm, &data);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (km[[i, j]] - km[[j, i]]).abs() < 1e-10,
                    "Kernel matrix should be symmetric"
                );
            }
            assert!(
                (km[[i, i]] - 1.0).abs() < 1e-10,
                "Diagonal should be 1"
            );
        }
    }

    #[test]
    fn test_rolling_std() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_std(&data, 3);
        assert_eq!(result.len(), 3);
        // std of [1,2,3] = sqrt(2/3) ≈ 0.8165
        assert!((result[0] - 0.8165).abs() < 0.01);
    }

    #[test]
    fn test_label_regimes() {
        let vols = vec![0.1, 0.5, 0.2, 0.8, 0.3];
        let labels = label_regimes(&vols);
        // median = 0.3; values >= 0.3 get +1, below get -1
        assert_eq!(labels, vec![-1.0, 1.0, -1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_normalize_feature() {
        let values = vec![0.0, 5.0, 10.0];
        let normed = normalize_feature(&values);
        assert!((normed[0] - 0.0).abs() < 1e-10);
        assert!((normed[1] - PI / 2.0).abs() < 1e-10);
        assert!((normed[2] - PI).abs() < 1e-10);
    }

    #[test]
    fn test_angle_encoding_closed_form() {
        // For angle encoding, K(x,y) = prod_i cos^2((x_i - y_i)/2)
        let fm = QuantumFeatureMap::new(3);
        let x = vec![0.5, 1.0, 1.5];
        let y = vec![0.8, 0.7, 1.2];
        let sx = fm.encode(&x);
        let sy = fm.encode(&y);
        let k_sim = fidelity_kernel(&sx, &sy);

        let k_formula: f64 = (0..3)
            .map(|i| ((x[i] - y[i]) / 2.0).cos().powi(2))
            .product();

        assert!(
            (k_sim - k_formula).abs() < 1e-10,
            "Simulation {} should match closed-form {}",
            k_sim,
            k_formula
        );
    }
}
