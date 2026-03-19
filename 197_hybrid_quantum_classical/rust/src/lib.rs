use std::f64::consts::PI;

use anyhow::{Context, Result};
use rand::Rng;
use serde::Deserialize;

// ============================================================================
// Complex number arithmetic for quantum state simulation
// ============================================================================

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

    pub fn norm_sq(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn mul(&self, other: &Complex) -> Complex {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    pub fn add(&self, other: &Complex) -> Complex {
        Complex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    pub fn scale(&self, s: f64) -> Complex {
        Complex {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

// ============================================================================
// Quantum circuit simulation
// ============================================================================

/// Computes the tensor (Kronecker) product of two statevectors.
pub fn tensor_product(a: &[Complex], b: &[Complex]) -> Vec<Complex> {
    let mut result = Vec::with_capacity(a.len() * b.len());
    for ai in a {
        for bj in b {
            result.push(ai.mul(bj));
        }
    }
    result
}

/// Applies RY(angle) gate to a single qubit state [alpha, beta].
fn ry_gate(angle: f64) -> [Complex; 4] {
    let c = (angle / 2.0).cos();
    let s = (angle / 2.0).sin();
    [
        Complex::new(c, 0.0),
        Complex::new(-s, 0.0),
        Complex::new(s, 0.0),
        Complex::new(c, 0.0),
    ]
}

/// Applies a 2x2 gate matrix to a single-qubit statevector.
pub fn apply_single_gate(state: &[Complex; 2], gate: &[Complex; 4]) -> [Complex; 2] {
    [
        gate[0].mul(&state[0]).add(&gate[1].mul(&state[1])),
        gate[2].mul(&state[0]).add(&gate[3].mul(&state[1])),
    ]
}

/// Applies a single-qubit gate to qubit `target` in a multi-qubit statevector.
fn apply_gate_to_qubit(statevector: &mut [Complex], num_qubits: usize, target: usize, gate: &[Complex; 4]) {
    let dim = 1 << num_qubits;
    let step = 1 << (num_qubits - 1 - target);
    let mut i = 0;
    while i < dim {
        for j in 0..step {
            let idx0 = i + j;
            let idx1 = i + j + step;
            let a = statevector[idx0];
            let b = statevector[idx1];
            statevector[idx0] = gate[0].mul(&a).add(&gate[1].mul(&b));
            statevector[idx1] = gate[2].mul(&a).add(&gate[3].mul(&b));
        }
        i += step * 2;
    }
}

/// Applies CNOT gate with `control` and `target` qubits.
fn apply_cnot(statevector: &mut [Complex], num_qubits: usize, control: usize, target: usize) {
    let dim = 1 << num_qubits;
    for i in 0..dim {
        let control_bit = (i >> (num_qubits - 1 - control)) & 1;
        let target_bit = (i >> (num_qubits - 1 - target)) & 1;
        if control_bit == 1 && target_bit == 0 {
            let j = i | (1 << (num_qubits - 1 - target));
            let tmp = statevector[i];
            statevector[i] = statevector[j];
            statevector[j] = tmp;
        }
    }
}

/// Computes <Z_i> expectation value for qubit i from the statevector.
fn expectation_z(statevector: &[Complex], num_qubits: usize, qubit: usize) -> f64 {
    let dim = 1 << num_qubits;
    let mut exp = 0.0;
    for i in 0..dim {
        let bit = (i >> (num_qubits - 1 - qubit)) & 1;
        let prob = statevector[i].norm_sq();
        if bit == 0 {
            exp += prob;
        } else {
            exp -= prob;
        }
    }
    exp
}

// ============================================================================
// Quantum Layer
// ============================================================================

/// A parameterized quantum circuit layer that transforms classical features
/// into quantum expectation values.
pub struct QuantumLayer {
    pub num_qubits: usize,
    pub num_layers: usize,
    /// Trainable parameters: num_layers * num_qubits angles for RY rotations
    pub params: Vec<f64>,
}

impl QuantumLayer {
    pub fn new(num_qubits: usize, num_layers: usize) -> Self {
        let num_params = num_layers * num_qubits;
        let mut rng = rand::thread_rng();
        let params: Vec<f64> = (0..num_params)
            .map(|_| rng.gen_range(0.0..2.0 * PI))
            .collect();
        Self {
            num_qubits,
            num_layers,
            params,
        }
    }

    /// Forward pass: encode features, apply variational layers, measure.
    /// Returns expectation values of Pauli-Z on each qubit.
    pub fn forward(&self, features: &[f64]) -> Vec<f64> {
        assert!(features.len() >= self.num_qubits);
        let dim = 1 << self.num_qubits;

        // Initialize |0...0> state
        let mut statevector = vec![Complex::zero(); dim];
        statevector[0] = Complex::new(1.0, 0.0);

        // Encode features using RY gates
        for q in 0..self.num_qubits {
            let gate = ry_gate(features[q]);
            apply_gate_to_qubit(&mut statevector, self.num_qubits, q, &gate);
        }

        // Apply variational layers
        for l in 0..self.num_layers {
            // Parameterized RY rotations
            for q in 0..self.num_qubits {
                let param_idx = l * self.num_qubits + q;
                let gate = ry_gate(self.params[param_idx]);
                apply_gate_to_qubit(&mut statevector, self.num_qubits, q, &gate);
            }
            // Entangling CNOT ladder
            for q in 0..(self.num_qubits - 1) {
                apply_cnot(&mut statevector, self.num_qubits, q, q + 1);
            }
        }

        // Measure expectation values
        (0..self.num_qubits)
            .map(|q| expectation_z(&statevector, self.num_qubits, q))
            .collect()
    }
}

// ============================================================================
// Classical Preprocessing Layer
// ============================================================================

/// Normalizes features to [0, pi] range using min-max scaling.
pub struct ClassicalPreprocessor {
    pub num_features: usize,
    pub min_vals: Vec<f64>,
    pub max_vals: Vec<f64>,
    pub fitted: bool,
}

impl ClassicalPreprocessor {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            min_vals: vec![f64::MAX; num_features],
            max_vals: vec![f64::MIN; num_features],
            fitted: false,
        }
    }

    /// Fit the preprocessor on training data to learn min/max values.
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        for sample in data {
            for (i, &val) in sample.iter().enumerate().take(self.num_features) {
                if val < self.min_vals[i] {
                    self.min_vals[i] = val;
                }
                if val > self.max_vals[i] {
                    self.max_vals[i] = val;
                }
            }
        }
        self.fitted = true;
    }

    /// Transform features to [0, pi] range.
    pub fn transform(&self, features: &[f64]) -> Vec<f64> {
        features
            .iter()
            .enumerate()
            .take(self.num_features)
            .map(|(i, &val)| {
                let range = self.max_vals[i] - self.min_vals[i];
                if range.abs() < 1e-10 {
                    PI / 2.0
                } else {
                    ((val - self.min_vals[i]) / range) * PI
                }
            })
            .collect()
    }
}

// ============================================================================
// Classical Postprocessing Layer
// ============================================================================

/// Linear classifier on quantum measurement results followed by sigmoid.
pub struct ClassicalPostprocessor {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl ClassicalPostprocessor {
    pub fn new(input_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights: Vec<f64> = (0..input_size)
            .map(|_| rng.gen_range(-0.5..0.5))
            .collect();
        let bias = rng.gen_range(-0.1..0.1);
        Self { weights, bias }
    }

    /// Forward pass: linear combination followed by sigmoid activation.
    pub fn forward(&self, inputs: &[f64]) -> f64 {
        let logit: f64 = inputs
            .iter()
            .zip(self.weights.iter())
            .map(|(x, w)| x * w)
            .sum::<f64>()
            + self.bias;
        sigmoid(logit)
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ============================================================================
// SPSA Optimizer
// ============================================================================

/// Simultaneous Perturbation Stochastic Approximation optimizer.
pub struct SPSAOptimizer {
    pub a: f64,
    pub c: f64,
    pub big_a: f64,
    pub alpha: f64,
    pub gamma: f64,
    pub iteration: usize,
}

impl SPSAOptimizer {
    pub fn new() -> Self {
        Self {
            a: 0.1,
            c: 0.1,
            big_a: 10.0,
            alpha: 0.602,
            gamma: 0.101,
            iteration: 0,
        }
    }

    /// Get current learning rate a_k.
    fn a_k(&self) -> f64 {
        self.a / (self.iteration as f64 + 1.0 + self.big_a).powf(self.alpha)
    }

    /// Get current perturbation magnitude c_k.
    fn c_k(&self) -> f64 {
        self.c / (self.iteration as f64 + 1.0).powf(self.gamma)
    }

    /// Perform one SPSA step on a parameter vector.
    /// `loss_fn` evaluates the loss for a given parameter vector.
    pub fn step<F>(&mut self, params: &mut Vec<f64>, loss_fn: F)
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut rng = rand::thread_rng();
        let ak = self.a_k();
        let ck = self.c_k();

        // Generate random perturbation vector
        let delta: Vec<f64> = (0..params.len())
            .map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 })
            .collect();

        // Evaluate loss at perturbed points
        let params_plus: Vec<f64> = params
            .iter()
            .zip(delta.iter())
            .map(|(p, d)| p + ck * d)
            .collect();
        let params_minus: Vec<f64> = params
            .iter()
            .zip(delta.iter())
            .map(|(p, d)| p - ck * d)
            .collect();

        let loss_plus = loss_fn(&params_plus);
        let loss_minus = loss_fn(&params_minus);

        // Estimate gradient and update
        for i in 0..params.len() {
            let grad = (loss_plus - loss_minus) / (2.0 * ck * delta[i]);
            params[i] -= ak * grad;
        }

        self.iteration += 1;
    }
}

impl Default for SPSAOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Hybrid Model
// ============================================================================

/// A complete hybrid quantum-classical model for trading signal prediction.
pub struct HybridModel {
    pub preprocessor: ClassicalPreprocessor,
    pub quantum_layer: QuantumLayer,
    pub postprocessor: ClassicalPostprocessor,
    pub optimizer: SPSAOptimizer,
}

impl HybridModel {
    pub fn new(num_features: usize, num_qubits: usize, num_layers: usize) -> Self {
        Self {
            preprocessor: ClassicalPreprocessor::new(num_features),
            quantum_layer: QuantumLayer::new(num_qubits, num_layers),
            postprocessor: ClassicalPostprocessor::new(num_qubits),
            optimizer: SPSAOptimizer::new(),
        }
    }

    /// Forward pass through the full pipeline.
    pub fn predict_single(&self, features: &[f64]) -> f64 {
        let normalized = self.preprocessor.transform(features);
        let quantum_out = self.quantum_layer.forward(&normalized);
        self.postprocessor.forward(&quantum_out)
    }

    /// Predict on a batch of samples.
    pub fn predict(&self, features: &[Vec<f64>]) -> Vec<f64> {
        features
            .iter()
            .map(|f| self.predict_single(f))
            .collect()
    }

    /// Compute binary cross-entropy loss over a dataset.
    pub fn compute_loss(&self, features: &[Vec<f64>], labels: &[f64], quantum_params: &[f64]) -> f64 {
        // Temporarily use provided quantum params
        let temp_quantum = QuantumLayer {
            num_qubits: self.quantum_layer.num_qubits,
            num_layers: self.quantum_layer.num_layers,
            params: quantum_params.to_vec(),
        };

        let n = features.len() as f64;
        let mut loss = 0.0;
        for (feat, &label) in features.iter().zip(labels.iter()) {
            let normalized = self.preprocessor.transform(feat);
            let quantum_out = temp_quantum.forward(&normalized);
            let pred = self.postprocessor.forward(&quantum_out);
            let pred_clipped = pred.clamp(1e-7, 1.0 - 1e-7);
            loss -= label * pred_clipped.ln() + (1.0 - label) * (1.0 - pred_clipped).ln();
        }
        loss / n
    }

    /// Train the model on labeled data.
    pub fn train(&mut self, features: &[Vec<f64>], labels: &[f64], epochs: usize) {
        // Fit preprocessor
        self.preprocessor.fit(features);

        for epoch in 0..epochs {
            // Optimize quantum parameters via SPSA
            let feat_clone: Vec<Vec<f64>> = features.to_vec();
            let labels_clone: Vec<f64> = labels.to_vec();

            let preprocessor = ClassicalPreprocessor {
                num_features: self.preprocessor.num_features,
                min_vals: self.preprocessor.min_vals.clone(),
                max_vals: self.preprocessor.max_vals.clone(),
                fitted: true,
            };
            let num_qubits = self.quantum_layer.num_qubits;
            let num_layers = self.quantum_layer.num_layers;
            let post_weights = self.postprocessor.weights.clone();
            let post_bias = self.postprocessor.bias;

            let loss_fn = |params: &[f64]| -> f64 {
                let temp_quantum = QuantumLayer {
                    num_qubits,
                    num_layers,
                    params: params.to_vec(),
                };
                let n = feat_clone.len() as f64;
                let mut loss = 0.0;
                for (feat, &label) in feat_clone.iter().zip(labels_clone.iter()) {
                    let normalized = preprocessor.transform(feat);
                    let quantum_out = temp_quantum.forward(&normalized);
                    let logit: f64 = quantum_out
                        .iter()
                        .zip(post_weights.iter())
                        .map(|(x, w)| x * w)
                        .sum::<f64>()
                        + post_bias;
                    let pred = sigmoid(logit).clamp(1e-7, 1.0 - 1e-7);
                    loss -= label * pred.ln() + (1.0 - label) * (1.0 - pred).ln();
                }
                loss / n
            };

            self.optimizer
                .step(&mut self.quantum_layer.params, &loss_fn);

            // Also update classical postprocessor weights with simple gradient step
            self.update_classical_weights(features, labels);

            if epoch % 10 == 0 {
                let loss = loss_fn(&self.quantum_layer.params);
                let accuracy = self.evaluate_accuracy(features, labels);
                println!(
                    "Epoch {}: loss = {:.4}, accuracy = {:.2}%",
                    epoch,
                    loss,
                    accuracy * 100.0
                );
            }
        }
    }

    /// Simple gradient update for classical postprocessor weights.
    fn update_classical_weights(&mut self, features: &[Vec<f64>], labels: &[f64]) {
        let lr = 0.01;
        let n = features.len() as f64;
        let mut weight_grads = vec![0.0; self.postprocessor.weights.len()];
        let mut bias_grad = 0.0;

        for (feat, &label) in features.iter().zip(labels.iter()) {
            let normalized = self.preprocessor.transform(feat);
            let quantum_out = self.quantum_layer.forward(&normalized);
            let pred = self.postprocessor.forward(&quantum_out);
            let error = pred - label;

            for (i, &q) in quantum_out.iter().enumerate() {
                weight_grads[i] += error * q;
            }
            bias_grad += error;
        }

        for (w, g) in self.postprocessor.weights.iter_mut().zip(weight_grads.iter()) {
            *w -= lr * g / n;
        }
        self.postprocessor.bias -= lr * bias_grad / n;
    }

    /// Evaluate classification accuracy.
    pub fn evaluate_accuracy(&self, features: &[Vec<f64>], labels: &[f64]) -> f64 {
        let predictions = self.predict(features);
        let correct = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(&pred, &label)| {
                let predicted_class = if pred >= 0.5 { 1.0 } else { 0.0 };
                (predicted_class - label).abs() < 1e-7
            })
            .count();
        correct as f64 / labels.len() as f64
    }
}

// ============================================================================
// Pure Classical Model for comparison
// ============================================================================

/// A simple logistic regression model for baseline comparison.
pub struct ClassicalModel {
    pub preprocessor: ClassicalPreprocessor,
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl ClassicalModel {
    pub fn new(num_features: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            preprocessor: ClassicalPreprocessor::new(num_features),
            weights: (0..num_features)
                .map(|_| rng.gen_range(-0.5..0.5))
                .collect(),
            bias: rng.gen_range(-0.1..0.1),
        }
    }

    pub fn predict_single(&self, features: &[f64]) -> f64 {
        let normalized = self.preprocessor.transform(features);
        let logit: f64 = normalized
            .iter()
            .zip(self.weights.iter())
            .map(|(x, w)| x * w)
            .sum::<f64>()
            + self.bias;
        sigmoid(logit)
    }

    pub fn predict(&self, features: &[Vec<f64>]) -> Vec<f64> {
        features
            .iter()
            .map(|f| self.predict_single(f))
            .collect()
    }

    pub fn train(&mut self, features: &[Vec<f64>], labels: &[f64], epochs: usize) {
        self.preprocessor.fit(features);
        let lr = 0.01;

        for epoch in 0..epochs {
            let n = features.len() as f64;
            let mut weight_grads = vec![0.0; self.weights.len()];
            let mut bias_grad = 0.0;

            for (feat, &label) in features.iter().zip(labels.iter()) {
                let normalized = self.preprocessor.transform(feat);
                let logit: f64 = normalized
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(x, w)| x * w)
                    .sum::<f64>()
                    + self.bias;
                let pred = sigmoid(logit);
                let error = pred - label;

                for (i, &x) in normalized.iter().enumerate() {
                    weight_grads[i] += error * x;
                }
                bias_grad += error;
            }

            for (w, g) in self.weights.iter_mut().zip(weight_grads.iter()) {
                *w -= lr * g / n;
            }
            self.bias -= lr * bias_grad / n;

            if epoch % 10 == 0 {
                let accuracy = self.evaluate_accuracy(features, labels);
                println!(
                    "Classical Epoch {}: accuracy = {:.2}%",
                    epoch,
                    accuracy * 100.0
                );
            }
        }
    }

    pub fn evaluate_accuracy(&self, features: &[Vec<f64>], labels: &[f64]) -> f64 {
        let predictions = self.predict(features);
        let correct = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(&pred, &label)| {
                let predicted_class = if pred >= 0.5 { 1.0 } else { 0.0 };
                (predicted_class - label).abs() < 1e-7
            })
            .count();
        correct as f64 / labels.len() as f64
    }
}

// ============================================================================
// Bybit API Client
// ============================================================================

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

/// Client for fetching market data from the Bybit API.
pub struct BybitClient {
    base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch OHLCV kline data for a given symbol and interval.
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response: BybitResponse = reqwest::blocking::get(&url)
            .context("Failed to fetch from Bybit API")?
            .json()
            .context("Failed to parse Bybit response")?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: ret_code = {}", response.ret_code);
        }

        let mut candles: Vec<Candle> = response
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() >= 6 {
                    Some(Candle {
                        timestamp: row[0].parse().ok()?,
                        open: row[1].parse().ok()?,
                        high: row[2].parse().ok()?,
                        low: row[3].parse().ok()?,
                        close: row[4].parse().ok()?,
                        volume: row[5].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first; reverse to chronological order
        candles.reverse();
        Ok(candles)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Feature Engineering
// ============================================================================

/// Compute trading features from candle data.
/// Returns (features, labels) where each feature vector has 4 elements:
/// [log_return, volatility, rsi, sma_ratio]
/// Labels: 1.0 if next candle closes higher, 0.0 otherwise.
pub fn compute_features(candles: &[Candle]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = candles.len();
    if n < 25 {
        return (vec![], vec![]);
    }

    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();

    // Log returns
    let returns: Vec<f64> = (1..n).map(|i| (closes[i] / closes[i - 1]).ln()).collect();

    let mut features = Vec::new();
    let mut labels = Vec::new();

    // Start from index 20 to have enough history for all indicators
    let start = 20;
    for i in start..(n - 1) {
        // Log return
        let log_return = returns[i - 1];

        // Volatility: std dev of last 10 returns
        let vol_window = &returns[(i - 10)..i];
        let mean_ret: f64 = vol_window.iter().sum::<f64>() / vol_window.len() as f64;
        let volatility = (vol_window
            .iter()
            .map(|r| (r - mean_ret).powi(2))
            .sum::<f64>()
            / vol_window.len() as f64)
            .sqrt();

        // RSI (14-period)
        let rsi_window = &returns[(i - 14)..i];
        let gains: f64 = rsi_window.iter().filter(|&&r| r > 0.0).sum();
        let losses: f64 = rsi_window.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
        let avg_gain = gains / 14.0;
        let avg_loss = losses / 14.0;
        let rsi = if avg_loss < 1e-10 {
            1.0
        } else {
            let rs = avg_gain / avg_loss;
            1.0 - 1.0 / (1.0 + rs)
        };

        // SMA ratio (20-period)
        let sma_window = &closes[(i - 19)..=i];
        let sma = sma_window.iter().sum::<f64>() / sma_window.len() as f64;
        let sma_ratio = closes[i] / sma;

        features.push(vec![log_return, volatility, rsi, sma_ratio]);

        // Label: 1.0 if next close is higher
        let label = if closes[i + 1] > closes[i] { 1.0 } else { 0.0 };
        labels.push(label);
    }

    (features, labels)
}

/// Generate synthetic candle data for testing when API is unavailable.
pub fn generate_synthetic_candles(n: usize) -> Vec<Candle> {
    let mut rng = rand::thread_rng();
    let mut candles = Vec::with_capacity(n);
    let mut price = 50000.0;

    for i in 0..n {
        let change = rng.gen_range(-0.02..0.02);
        let open: f64 = price;
        let close: f64 = price * (1.0 + change);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100.0..10000.0);

        candles.push(Candle {
            timestamp: 1700000000000 + (i as u64) * 3600000,
            open,
            high,
            low,
            close,
            volume,
        });

        price = close;
    }

    candles
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_arithmetic() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let c = a.mul(&b);
        assert!((c.re - (-5.0)).abs() < 1e-10);
        assert!((c.im - 10.0).abs() < 1e-10);

        let d = a.add(&b);
        assert!((d.re - 4.0).abs() < 1e-10);
        assert!((d.im - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_product() {
        let a = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]; // |0>
        let b = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]; // |0>
        let result = tensor_product(&a, &b);
        assert_eq!(result.len(), 4);
        assert!((result[0].re - 1.0).abs() < 1e-10); // |00> amplitude = 1
        assert!((result[1].re).abs() < 1e-10);
        assert!((result[2].re).abs() < 1e-10);
        assert!((result[3].re).abs() < 1e-10);
    }

    #[test]
    fn test_ry_gate_zero_angle() {
        let gate = ry_gate(0.0);
        // RY(0) = Identity
        assert!((gate[0].re - 1.0).abs() < 1e-10);
        assert!((gate[3].re - 1.0).abs() < 1e-10);
        assert!((gate[1].re).abs() < 1e-10);
        assert!((gate[2].re).abs() < 1e-10);
    }

    #[test]
    fn test_ry_gate_pi() {
        let gate = ry_gate(PI);
        // RY(pi) should flip |0> to |1>
        let state = [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        let result = apply_single_gate(&state, &gate);
        assert!((result[0].re).abs() < 1e-10);
        assert!((result[1].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_layer_output_range() {
        let layer = QuantumLayer::new(4, 2);
        let features = vec![0.5, 1.0, 1.5, 2.0];
        let output = layer.forward(&features);
        assert_eq!(output.len(), 4);
        for &val in &output {
            assert!(val >= -1.0 && val <= 1.0, "Expectation value out of range: {}", val);
        }
    }

    #[test]
    fn test_classical_preprocessor() {
        let mut prep = ClassicalPreprocessor::new(3);
        let data = vec![
            vec![0.0, 10.0, -1.0],
            vec![1.0, 20.0, 1.0],
            vec![0.5, 15.0, 0.0],
        ];
        prep.fit(&data);

        let transformed = prep.transform(&[0.5, 15.0, 0.0]);
        assert_eq!(transformed.len(), 3);
        // 0.5 is midpoint of [0, 1] -> pi/2
        assert!((transformed[0] - PI / 2.0).abs() < 1e-10);
        // 15.0 is midpoint of [10, 20] -> pi/2
        assert!((transformed[1] - PI / 2.0).abs() < 1e-10);
        // 0.0 is midpoint of [-1, 1] -> pi/2
        assert!((transformed[2] - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_classical_postprocessor_output_range() {
        let post = ClassicalPostprocessor::new(4);
        let inputs = vec![0.5, -0.3, 0.8, -0.1];
        let output = post.forward(&inputs);
        assert!(output >= 0.0 && output <= 1.0, "Sigmoid output out of range: {}", output);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(100.0) > 0.99);
        assert!(sigmoid(-100.0) < 0.01);
    }

    #[test]
    fn test_hybrid_model_forward() {
        let mut model = HybridModel::new(4, 4, 2);
        let data = vec![
            vec![0.01, 0.02, 0.5, 1.0],
            vec![-0.01, 0.03, 0.3, 0.98],
        ];
        model.preprocessor.fit(&data);
        let pred = model.predict_single(&data[0]);
        assert!(pred >= 0.0 && pred <= 1.0, "Prediction out of range: {}", pred);
    }

    #[test]
    fn test_hybrid_model_train() {
        let mut model = HybridModel::new(4, 4, 1);
        let features = vec![
            vec![0.01, 0.02, 0.6, 1.01],
            vec![-0.01, 0.03, 0.4, 0.99],
            vec![0.02, 0.01, 0.7, 1.02],
            vec![-0.02, 0.04, 0.3, 0.98],
        ];
        let labels = vec![1.0, 0.0, 1.0, 0.0];
        // Just verify it runs without panic
        model.train(&features, &labels, 5);
    }

    #[test]
    fn test_spsa_optimizer() {
        let mut optimizer = SPSAOptimizer::new();
        let mut params = vec![1.0, 2.0, 3.0];
        let loss_fn = |p: &[f64]| -> f64 {
            p.iter().map(|x| x * x).sum::<f64>()
        };
        let initial_loss = loss_fn(&params);
        for _ in 0..50 {
            optimizer.step(&mut params, &loss_fn);
        }
        let final_loss = loss_fn(&params);
        assert!(final_loss < initial_loss, "SPSA should reduce loss");
    }

    #[test]
    fn test_cnot_gate() {
        let num_qubits = 2;
        let dim = 1 << num_qubits;
        // Start with |10> state
        let mut statevector = vec![Complex::zero(); dim];
        statevector[2] = Complex::new(1.0, 0.0); // |10>

        apply_cnot(&mut statevector, num_qubits, 0, 1);

        // Should become |11>
        assert!((statevector[3].norm_sq() - 1.0).abs() < 1e-10);
        assert!((statevector[2].norm_sq()).abs() < 1e-10);
    }

    #[test]
    fn test_expectation_z() {
        let num_qubits = 1;
        // |0> state: Z expectation = +1
        let state_0 = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        assert!((expectation_z(&state_0, num_qubits, 0) - 1.0).abs() < 1e-10);

        // |1> state: Z expectation = -1
        let state_1 = vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)];
        assert!((expectation_z(&state_1, num_qubits, 0) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_compute_features_synthetic() {
        let candles = generate_synthetic_candles(100);
        let (features, labels) = compute_features(&candles);
        assert!(!features.is_empty());
        assert_eq!(features.len(), labels.len());
        assert_eq!(features[0].len(), 4);
        for label in &labels {
            assert!(*label == 0.0 || *label == 1.0);
        }
    }

    #[test]
    fn test_generate_synthetic_candles() {
        let candles = generate_synthetic_candles(50);
        assert_eq!(candles.len(), 50);
        for candle in &candles {
            assert!(candle.high >= candle.open.max(candle.close));
            assert!(candle.low <= candle.open.min(candle.close));
            assert!(candle.volume > 0.0);
        }
    }

    #[test]
    fn test_classical_model_train() {
        let mut model = ClassicalModel::new(4);
        let features = vec![
            vec![0.01, 0.02, 0.6, 1.01],
            vec![-0.01, 0.03, 0.4, 0.99],
            vec![0.02, 0.01, 0.7, 1.02],
            vec![-0.02, 0.04, 0.3, 0.98],
        ];
        let labels = vec![1.0, 0.0, 1.0, 0.0];
        model.train(&features, &labels, 5);
        let pred = model.predict_single(&features[0]);
        assert!(pred >= 0.0 && pred <= 1.0);
    }
}
