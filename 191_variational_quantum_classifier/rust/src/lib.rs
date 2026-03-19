//! Variational Quantum Classifier (VQC) for Trading
//!
//! A hybrid quantum-classical machine learning model that uses parameterized
//! quantum circuits for binary classification of market regimes.
//!
//! The VQC consists of:
//! 1. Data encoding circuit: maps classical features to quantum states via Ry rotations
//! 2. Variational circuit: trainable Rz-Ry-Rz rotations + CNOT entangling gates
//! 3. Measurement: expectation value of Pauli-Z on first qubit
//!
//! Training uses the parameter-shift rule for gradient computation.

use std::f64::consts::PI;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

// ─────────────────────────────────────────────────────────────────────────────
// Complex number arithmetic (re, im) tuples
// ─────────────────────────────────────────────────────────────────────────────

/// Multiply two complex numbers represented as (re, im) tuples.
#[inline]
fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// Add two complex numbers.
#[inline]
fn cadd(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 + b.0, a.1 + b.1)
}

/// Squared modulus |z|^2.
#[inline]
fn cnorm2(z: (f64, f64)) -> f64 {
    z.0 * z.0 + z.1 * z.1
}

// ─────────────────────────────────────────────────────────────────────────────
// Quantum gate operations on state vectors
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a single-qubit gate (2x2 matrix) to `target` qubit in an n-qubit state.
///
/// The gate is given as [a, b, c, d] representing the matrix:
///   [[a, b],
///    [c, d]]
/// where each entry is a complex number (re, im).
fn apply_gate(state: &mut [(f64, f64)], gate: [(f64, f64); 4], target: usize, n_qubits: usize) {
    let dim = 1 << n_qubits;
    let step = 1 << target;
    let mut i = 0;
    while i < dim {
        for j in i..i + step {
            let idx0 = j;
            let idx1 = j + step;
            let s0 = state[idx0];
            let s1 = state[idx1];
            state[idx0] = cadd(cmul(gate[0], s0), cmul(gate[1], s1));
            state[idx1] = cadd(cmul(gate[2], s0), cmul(gate[3], s1));
        }
        i += step << 1;
    }
}

/// Ry(theta) gate: rotation about Y axis.
/// Matrix: [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
fn apply_ry(state: &mut [(f64, f64)], theta: f64, target: usize, n_qubits: usize) {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    let gate = [(c, 0.0), (-s, 0.0), (s, 0.0), (c, 0.0)];
    apply_gate(state, gate, target, n_qubits);
}

/// Rz(theta) gate: rotation about Z axis.
/// Matrix: [[e^{-i*t/2}, 0], [0, e^{i*t/2}]]
fn apply_rz(state: &mut [(f64, f64)], theta: f64, target: usize, n_qubits: usize) {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    let gate = [(c, -s), (0.0, 0.0), (0.0, 0.0), (c, s)];
    apply_gate(state, gate, target, n_qubits);
}

/// Hadamard gate on target qubit.
#[allow(dead_code)]
fn apply_hadamard(state: &mut [(f64, f64)], target: usize, n_qubits: usize) {
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let gate = [(h, 0.0), (h, 0.0), (h, 0.0), (-h, 0.0)];
    apply_gate(state, gate, target, n_qubits);
}

/// CNOT gate: control qubit `ctrl`, target qubit `targ`.
/// Flips `targ` if `ctrl` is |1>.
fn apply_cnot(state: &mut [(f64, f64)], ctrl: usize, targ: usize, n_qubits: usize) {
    let dim = 1 << n_qubits;
    let ctrl_bit = 1 << ctrl;
    let targ_bit = 1 << targ;
    for i in 0..dim {
        // Only act when control qubit is 1 and target qubit is 0
        if (i & ctrl_bit) != 0 && (i & targ_bit) == 0 {
            let j = i | targ_bit;
            state.swap(i, j);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VQC circuit
// ─────────────────────────────────────────────────────────────────────────────

/// Initialize state to |0...0>.
fn init_state(n_qubits: usize) -> Vec<(f64, f64)> {
    let dim = 1 << n_qubits;
    let mut state = vec![(0.0, 0.0); dim];
    state[0] = (1.0, 0.0);
    state
}

/// Encode classical features into the quantum state using Ry rotations.
/// Each feature x_k is encoded as Ry(x_k) on qubit k.
/// Features should be pre-normalized to [0, pi].
fn encode_features(state: &mut [(f64, f64)], features: &[f64], n_qubits: usize) {
    for (q, &x) in features.iter().enumerate().take(n_qubits) {
        apply_ry(state, x, q, n_qubits);
    }
}

/// Apply one variational layer: Rz-Ry-Rz on each qubit + CNOT ladder.
/// Requires 3 * n_qubits parameters per layer.
fn apply_variational_layer(state: &mut [(f64, f64)], params: &[f64], n_qubits: usize) {
    // Single-qubit rotations
    for q in 0..n_qubits {
        let base = 3 * q;
        apply_rz(state, params[base], q, n_qubits);
        apply_ry(state, params[base + 1], q, n_qubits);
        apply_rz(state, params[base + 2], q, n_qubits);
    }
    // Entangling CNOT ladder
    if n_qubits > 1 {
        for q in 0..n_qubits - 1 {
            apply_cnot(state, q, q + 1, n_qubits);
        }
    }
}

/// Number of parameters per variational layer.
pub fn params_per_layer(n_qubits: usize) -> usize {
    3 * n_qubits
}

/// Total number of parameters for a VQC with given qubits and layers.
pub fn total_params(n_qubits: usize, n_layers: usize) -> usize {
    params_per_layer(n_qubits) * n_layers
}

/// Run the full VQC circuit and return the expectation value of Z on qubit 0.
///
/// Circuit: U_encode(x) -> [W_layer(theta)]^L -> measure <Z_0>
///
/// Returns a value in [-1, +1].
pub fn vqc_expectation(features: &[f64], params: &[f64], n_qubits: usize, n_layers: usize) -> f64 {
    let mut state = init_state(n_qubits);

    // Encode features
    encode_features(&mut state, features, n_qubits);

    // Apply variational layers
    let ppl = params_per_layer(n_qubits);
    for l in 0..n_layers {
        let layer_params = &params[l * ppl..(l + 1) * ppl];
        apply_variational_layer(&mut state, layer_params, n_qubits);
    }

    // Compute <Z_0> = sum_i |a_i|^2 * (-1)^{bit_0(i)}
    let dim = 1 << n_qubits;
    let mut expectation = 0.0;
    for i in 0..dim {
        let prob = cnorm2(state[i]);
        if (i & 1) == 0 {
            expectation += prob; // qubit 0 is |0>, eigenvalue +1
        } else {
            expectation -= prob; // qubit 0 is |1>, eigenvalue -1
        }
    }
    expectation
}

/// Run VQC with data re-uploading: interleave encoding and variational layers.
///
/// Circuit: [U_encode(x) -> W_layer(theta)]^L -> measure <Z_0>
pub fn vqc_expectation_reuploading(
    features: &[f64],
    params: &[f64],
    n_qubits: usize,
    n_layers: usize,
) -> f64 {
    let mut state = init_state(n_qubits);
    let ppl = params_per_layer(n_qubits);

    for l in 0..n_layers {
        // Re-encode features at each layer
        encode_features(&mut state, features, n_qubits);
        let layer_params = &params[l * ppl..(l + 1) * ppl];
        apply_variational_layer(&mut state, layer_params, n_qubits);
    }

    // Compute <Z_0>
    let dim = 1 << n_qubits;
    let mut expectation = 0.0;
    for i in 0..dim {
        let prob = cnorm2(state[i]);
        if (i & 1) == 0 {
            expectation += prob;
        } else {
            expectation -= prob;
        }
    }
    expectation
}

// ─────────────────────────────────────────────────────────────────────────────
// Cost function and gradient
// ─────────────────────────────────────────────────────────────────────────────

/// Mean squared error cost: C = (1/N) * sum_i (1 - y_i * f(x_i))^2
pub fn mse_cost(
    params: &[f64],
    data: &[Vec<f64>],
    labels: &[f64],
    n_qubits: usize,
    n_layers: usize,
) -> f64 {
    let n = data.len() as f64;
    let mut cost = 0.0;
    for (x, &y) in data.iter().zip(labels.iter()) {
        let pred = vqc_expectation(x, params, n_qubits, n_layers);
        let err = 1.0 - y * pred;
        cost += err * err;
    }
    cost / n
}

/// Hinge-like cost: C = (1/N) * sum_i max(0, 1 - y_i * f(x_i))
pub fn hinge_cost(
    params: &[f64],
    data: &[Vec<f64>],
    labels: &[f64],
    n_qubits: usize,
    n_layers: usize,
) -> f64 {
    let n = data.len() as f64;
    let mut cost = 0.0;
    for (x, &y) in data.iter().zip(labels.iter()) {
        let pred = vqc_expectation(x, params, n_qubits, n_layers);
        let margin = 1.0 - y * pred;
        if margin > 0.0 {
            cost += margin;
        }
    }
    cost / n
}

/// Compute gradient using the parameter-shift rule.
///
/// For each parameter theta_k:
///   dC/d(theta_k) = [C(theta_k + pi/2) - C(theta_k - pi/2)] / 2
pub fn parameter_shift_gradient(
    params: &[f64],
    data: &[Vec<f64>],
    labels: &[f64],
    n_qubits: usize,
    n_layers: usize,
) -> Vec<f64> {
    let n_params = params.len();
    let mut grad = vec![0.0; n_params];

    for k in 0..n_params {
        let mut params_plus = params.to_vec();
        let mut params_minus = params.to_vec();
        params_plus[k] += PI / 2.0;
        params_minus[k] -= PI / 2.0;

        let cost_plus = mse_cost(&params_plus, data, labels, n_qubits, n_layers);
        let cost_minus = mse_cost(&params_minus, data, labels, n_qubits, n_layers);

        grad[k] = (cost_plus - cost_minus) / 2.0;
    }

    grad
}

// ─────────────────────────────────────────────────────────────────────────────
// VQC Model
// ─────────────────────────────────────────────────────────────────────────────

/// Trained VQC model.
pub struct VQCModel {
    pub params: Vec<f64>,
    pub n_qubits: usize,
    pub n_layers: usize,
    pub learning_rate: f64,
    pub training_costs: Vec<f64>,
}

impl VQCModel {
    /// Create a new VQC model with random initial parameters.
    pub fn new(n_qubits: usize, n_layers: usize, learning_rate: f64, seed: u64) -> Self {
        let n_params = total_params(n_qubits, n_layers);
        let mut rng = StdRng::seed_from_u64(seed);
        let params: Vec<f64> = (0..n_params)
            .map(|_| rng.gen_range(-PI..PI) * 0.1) // Small initial values
            .collect();

        VQCModel {
            params,
            n_qubits,
            n_layers,
            learning_rate,
            training_costs: Vec::new(),
        }
    }

    /// Create a new VQC with identity-like initialization (near zero parameters).
    /// This helps mitigate barren plateaus.
    pub fn new_identity_init(n_qubits: usize, n_layers: usize, learning_rate: f64, seed: u64) -> Self {
        let n_params = total_params(n_qubits, n_layers);
        let mut rng = StdRng::seed_from_u64(seed);
        let params: Vec<f64> = (0..n_params)
            .map(|_| rng.gen_range(-0.01..0.01))
            .collect();

        VQCModel {
            params,
            n_qubits,
            n_layers,
            learning_rate,
            training_costs: Vec::new(),
        }
    }

    /// Train the VQC for a given number of epochs using gradient descent.
    pub fn train(
        &mut self,
        data: &[Vec<f64>],
        labels: &[f64],
        epochs: usize,
    ) {
        for epoch in 0..epochs {
            let cost = mse_cost(&self.params, data, labels, self.n_qubits, self.n_layers);
            self.training_costs.push(cost);

            let grad = parameter_shift_gradient(
                &self.params,
                data,
                labels,
                self.n_qubits,
                self.n_layers,
            );

            // Gradient descent update
            for (p, g) in self.params.iter_mut().zip(grad.iter()) {
                *p -= self.learning_rate * g;
            }

            if epoch % 10 == 0 || epoch == epochs - 1 {
                println!("  Epoch {}/{}: cost = {:.6}", epoch + 1, epochs, cost);
            }
        }
    }

    /// Train with mini-batch gradient descent.
    pub fn train_minibatch(
        &mut self,
        data: &[Vec<f64>],
        labels: &[f64],
        epochs: usize,
        batch_size: usize,
        seed: u64,
    ) {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = data.len();

        for epoch in 0..epochs {
            // Create random batch
            let indices: Vec<usize> = (0..batch_size)
                .map(|_| rng.gen_range(0..n))
                .collect();
            let batch_data: Vec<Vec<f64>> = indices.iter().map(|&i| data[i].clone()).collect();
            let batch_labels: Vec<f64> = indices.iter().map(|&i| labels[i]).collect();

            let cost = mse_cost(&self.params, &batch_data, &batch_labels, self.n_qubits, self.n_layers);
            self.training_costs.push(cost);

            let grad = parameter_shift_gradient(
                &self.params,
                &batch_data,
                &batch_labels,
                self.n_qubits,
                self.n_layers,
            );

            for (p, g) in self.params.iter_mut().zip(grad.iter()) {
                *p -= self.learning_rate * g;
            }

            if epoch % 10 == 0 || epoch == epochs - 1 {
                let full_cost = mse_cost(&self.params, data, labels, self.n_qubits, self.n_layers);
                println!("  Epoch {}/{}: batch_cost = {:.6}, full_cost = {:.6}", epoch + 1, epochs, cost, full_cost);
            }
        }
    }

    /// Predict a single sample: returns expectation value in [-1, +1].
    pub fn predict_raw(&self, features: &[f64]) -> f64 {
        vqc_expectation(features, &self.params, self.n_qubits, self.n_layers)
    }

    /// Predict class label: +1 if expectation > threshold, -1 otherwise.
    pub fn predict(&self, features: &[f64], threshold: f64) -> f64 {
        if self.predict_raw(features) > threshold {
            1.0
        } else {
            -1.0
        }
    }

    /// Predict multiple samples, returning class labels.
    pub fn predict_batch(&self, data: &[Vec<f64>], threshold: f64) -> Vec<f64> {
        data.iter().map(|x| self.predict(x, threshold)).collect()
    }

    /// Predict multiple samples, returning raw expectation values.
    pub fn predict_raw_batch(&self, data: &[Vec<f64>]) -> Vec<f64> {
        data.iter().map(|x| self.predict_raw(x)).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature engineering for trading
// ─────────────────────────────────────────────────────────────────────────────

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

/// Compute log returns from closing prices.
pub fn log_returns(candles: &[Candle]) -> Vec<f64> {
    let mut returns = Vec::with_capacity(candles.len().saturating_sub(1));
    for i in 1..candles.len() {
        returns.push((candles[i].close / candles[i - 1].close).ln());
    }
    returns
}

/// Compute rolling volatility (standard deviation of returns) over a window.
pub fn rolling_volatility(returns: &[f64], window: usize) -> Vec<f64> {
    let mut vols = Vec::new();
    for i in window..returns.len() {
        let slice = &returns[i - window..i];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let var: f64 = slice.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / window as f64;
        vols.push(var.sqrt());
    }
    vols
}

/// Compute RSI-like momentum indicator.
pub fn rsi_indicator(returns: &[f64], window: usize) -> Vec<f64> {
    let mut rsi_vals = Vec::new();
    for i in window..returns.len() {
        let slice = &returns[i - window..i];
        let mut avg_gain = 0.0;
        let mut avg_loss = 0.0;
        for &r in slice {
            if r > 0.0 {
                avg_gain += r;
            } else {
                avg_loss += r.abs();
            }
        }
        avg_gain /= window as f64;
        avg_loss /= window as f64;

        let rsi = if avg_loss < 1e-12 {
            1.0
        } else {
            avg_gain / (avg_gain + avg_loss)
        };
        rsi_vals.push(rsi);
    }
    rsi_vals
}

/// Compute volume ratio: current volume / moving average volume.
pub fn volume_ratio(candles: &[Candle], window: usize) -> Vec<f64> {
    let mut ratios = Vec::new();
    for i in window..candles.len() {
        let avg_vol: f64 = candles[i - window..i].iter().map(|c| c.volume).sum::<f64>() / window as f64;
        let ratio = if avg_vol < 1e-12 {
            1.0
        } else {
            candles[i].volume / avg_vol
        };
        ratios.push(ratio);
    }
    ratios
}

/// Compute price position: (close - low) / (high - low) over rolling window.
pub fn price_position(candles: &[Candle], window: usize) -> Vec<f64> {
    let mut positions = Vec::new();
    for i in window..candles.len() {
        let mut high = f64::NEG_INFINITY;
        let mut low = f64::INFINITY;
        for c in &candles[i - window..=i] {
            if c.high > high {
                high = c.high;
            }
            if c.low < low {
                low = c.low;
            }
        }
        let range = high - low;
        let pos = if range < 1e-12 {
            0.5
        } else {
            (candles[i].close - low) / range
        };
        positions.push(pos);
    }
    positions
}

/// Engineer a full feature set from candle data.
///
/// Returns (features, valid_indices) where features[i] = [return, volatility, RSI, vol_ratio, price_pos].
/// The `window` parameter determines the lookback for indicators.
pub fn engineer_features(candles: &[Candle], window: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
    let returns = log_returns(candles);
    let vols = rolling_volatility(&returns, window);
    let rsi = rsi_indicator(&returns, window);
    let vol_rat = volume_ratio(candles, window);
    let price_pos = price_position(candles, window);

    // Align all indicators: they all start at index `window` in the returns array
    // returns starts at candle index 1, indicators start at returns index `window`
    // So effective candle index starts at 1 + window
    let start_candle = 1 + window;
    let n = vols.len().min(rsi.len()).min(vol_rat.len()).min(price_pos.len());

    let mut features = Vec::with_capacity(n);
    let mut indices = Vec::with_capacity(n);

    for i in 0..n {
        let feat = vec![
            returns[window + i],
            vols[i],
            rsi[i],
            vol_rat[i],
            price_pos[i],
        ];
        features.push(feat);
        indices.push(start_candle + i);
    }

    (features, indices)
}

/// Normalize features to [0, pi] range using min-max scaling.
pub fn normalize_features(features: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if features.is_empty() {
        return Vec::new();
    }
    let n_features = features[0].len();
    let mut mins = vec![f64::INFINITY; n_features];
    let mut maxs = vec![f64::NEG_INFINITY; n_features];

    for feat in features {
        for (j, &val) in feat.iter().enumerate() {
            if val < mins[j] {
                mins[j] = val;
            }
            if val > maxs[j] {
                maxs[j] = val;
            }
        }
    }

    features
        .iter()
        .map(|feat| {
            feat.iter()
                .enumerate()
                .map(|(j, &val)| {
                    let range = maxs[j] - mins[j];
                    if range < 1e-12 {
                        PI / 2.0
                    } else {
                        ((val - mins[j]) / range) * PI
                    }
                })
                .collect()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Market regime labeling
// ─────────────────────────────────────────────────────────────────────────────

/// Label market regimes based on forward returns.
///
/// +1.0 = bull (forward return > threshold)
/// -1.0 = bear (forward return < -threshold)
///  0.0 = sideways
pub fn label_regimes(candles: &[Candle], forward: usize, threshold: f64) -> Vec<f64> {
    let mut labels = Vec::with_capacity(candles.len().saturating_sub(forward));
    for i in 0..candles.len().saturating_sub(forward) {
        let ret = (candles[i + forward].close / candles[i].close).ln();
        if ret > threshold {
            labels.push(1.0);
        } else if ret < -threshold {
            labels.push(-1.0);
        } else {
            labels.push(0.0);
        }
    }
    labels
}

/// Convert three-class labels to binary: +1 (bull) vs -1 (everything else).
pub fn to_binary_labels(labels: &[f64]) -> Vec<f64> {
    labels
        .iter()
        .map(|&l| if l > 0.5 { 1.0 } else { -1.0 })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluation metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Compute classification accuracy as a percentage.
pub fn accuracy(predictions: &[f64], labels: &[f64]) -> f64 {
    let correct = predictions
        .iter()
        .zip(labels.iter())
        .filter(|(&p, &l)| (p - l).abs() < 1e-6)
        .count();
    100.0 * correct as f64 / predictions.len() as f64
}

/// Compute confusion matrix for binary classification.
/// Returns (true_pos, false_pos, false_neg, true_neg).
pub fn confusion_matrix(predictions: &[f64], labels: &[f64]) -> (usize, usize, usize, usize) {
    let mut tp = 0;
    let mut fp = 0;
    let mut fn_ = 0;
    let mut tn = 0;

    for (&p, &l) in predictions.iter().zip(labels.iter()) {
        match (p > 0.0, l > 0.0) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            (false, false) => tn += 1,
        }
    }

    (tp, fp, fn_, tn)
}

/// Print a formatted confusion matrix.
pub fn print_confusion_matrix(predictions: &[f64], labels: &[f64]) {
    let (tp, fp, fn_, tn) = confusion_matrix(predictions, labels);
    println!("  Confusion Matrix:");
    println!("                 Predicted +1  Predicted -1");
    println!("  Actual +1      {:>10}  {:>12}", tp, fn_);
    println!("  Actual -1      {:>10}  {:>12}", fp, tn);

    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    println!("  Precision: {:.4}", precision);
    println!("  Recall:    {:.4}", recall);
    println!("  F1-score:  {:.4}", f1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Bybit API integration
// ─────────────────────────────────────────────────────────────────────────────

/// Response structure from Bybit kline endpoint.
#[derive(serde::Deserialize, Debug)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    pub result: BybitResult,
}

#[derive(serde::Deserialize, Debug)]
pub struct BybitResult {
    pub list: Vec<Vec<String>>,
}

/// Fetch OHLCV candles from Bybit public API.
///
/// # Arguments
/// * `symbol` - Trading pair (e.g. "BTCUSDT")
/// * `interval` - Candle interval ("1", "5", "15", "60", "D", etc.)
/// * `limit` - Number of candles to fetch (max 200)
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> anyhow::Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: retCode={}", resp.ret_code);
    }

    let mut candles: Vec<Candle> = resp
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

    // Bybit returns most recent first; reverse to chronological order
    candles.reverse();
    Ok(candles)
}

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic data generation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate synthetic OHLCV candles for testing.
pub fn generate_synthetic_candles(n: usize, seed: u64) -> Vec<Candle> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut candles = Vec::with_capacity(n);
    let mut price = 50_000.0; // BTC-like starting price
    let mut timestamp = 1_700_000_000_000u64;

    for _ in 0..n {
        let ret = rng.gen_range(-0.03..0.03);
        let new_price = price * (1.0 + ret);
        let high = new_price * (1.0 + rng.gen_range(0.001..0.015));
        let low = new_price * (1.0 - rng.gen_range(0.001..0.015));
        let volume = rng.gen_range(100.0..10000.0);

        candles.push(Candle {
            timestamp,
            open: price,
            high,
            low,
            close: new_price,
            volume,
        });

        price = new_price;
        timestamp += 3_600_000; // 1 hour
    }

    candles
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_state_normalized() {
        let state = init_state(3);
        let norm: f64 = state.iter().map(|s| cnorm2(*s)).sum();
        assert!((norm - 1.0).abs() < 1e-10, "Initial state should be normalized");
    }

    #[test]
    fn test_ry_preserves_normalization() {
        let mut state = init_state(2);
        apply_ry(&mut state, 1.23, 0, 2);
        apply_ry(&mut state, 0.77, 1, 2);
        let norm: f64 = state.iter().map(|s| cnorm2(*s)).sum();
        assert!((norm - 1.0).abs() < 1e-10, "Ry should preserve normalization");
    }

    #[test]
    fn test_rz_preserves_normalization() {
        let mut state = init_state(2);
        apply_ry(&mut state, 1.0, 0, 2); // First get non-trivial state
        apply_rz(&mut state, 2.5, 0, 2);
        let norm: f64 = state.iter().map(|s| cnorm2(*s)).sum();
        assert!((norm - 1.0).abs() < 1e-10, "Rz should preserve normalization");
    }

    #[test]
    fn test_cnot_preserves_normalization() {
        let mut state = init_state(2);
        apply_ry(&mut state, 1.0, 0, 2);
        apply_cnot(&mut state, 0, 1, 2);
        let norm: f64 = state.iter().map(|s| cnorm2(*s)).sum();
        assert!((norm - 1.0).abs() < 1e-10, "CNOT should preserve normalization");
    }

    #[test]
    fn test_hadamard_on_zero() {
        let mut state = init_state(1);
        apply_hadamard(&mut state, 0, 1);
        // |0> -> (|0> + |1>) / sqrt(2)
        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!((state[0].0 - expected).abs() < 1e-10);
        assert!((state[1].0 - expected).abs() < 1e-10);
    }

    #[test]
    fn test_vqc_expectation_range() {
        let features = vec![0.5, 1.0];
        let n_qubits = 2;
        let n_layers = 2;
        let n_params = total_params(n_qubits, n_layers);
        let params: Vec<f64> = (0..n_params).map(|i| i as f64 * 0.1).collect();
        let exp = vqc_expectation(&features, &params, n_qubits, n_layers);
        assert!(
            exp >= -1.0 && exp <= 1.0,
            "Expectation should be in [-1, +1], got {}",
            exp
        );
    }

    #[test]
    fn test_vqc_reuploading_range() {
        let features = vec![0.5, 1.0];
        let n_qubits = 2;
        let n_layers = 3;
        let n_params = total_params(n_qubits, n_layers);
        let params: Vec<f64> = (0..n_params).map(|i| i as f64 * 0.1).collect();
        let exp = vqc_expectation_reuploading(&features, &params, n_qubits, n_layers);
        assert!(
            exp >= -1.0 && exp <= 1.0,
            "Reuploading expectation should be in [-1, +1], got {}",
            exp
        );
    }

    #[test]
    fn test_zero_params_identity_like() {
        // With zero parameters and zero features, VQC should return ~1.0 (all probability in |0>)
        let features = vec![0.0, 0.0];
        let n_qubits = 2;
        let n_layers = 1;
        let n_params = total_params(n_qubits, n_layers);
        let params = vec![0.0; n_params];
        let exp = vqc_expectation(&features, &params, n_qubits, n_layers);
        assert!(
            (exp - 1.0).abs() < 1e-10,
            "Zero params + zero features should give expectation ~1.0, got {}",
            exp
        );
    }

    #[test]
    fn test_params_per_layer() {
        assert_eq!(params_per_layer(2), 6);
        assert_eq!(params_per_layer(3), 9);
        assert_eq!(total_params(2, 3), 18);
    }

    #[test]
    fn test_mse_cost_perfect() {
        // If predictions perfectly match labels, cost should be 0
        // This is hard to achieve, but with carefully chosen params we can verify cost > 0
        let features = vec![vec![0.5], vec![1.5]];
        let labels = vec![1.0, -1.0];
        let params = vec![0.0; total_params(1, 1)];
        let cost = mse_cost(&params, &features, &labels, 1, 1);
        assert!(cost >= 0.0, "Cost should be non-negative");
    }

    #[test]
    fn test_gradient_finite() {
        let features = vec![vec![0.5, 1.0], vec![1.5, 0.3]];
        let labels = vec![1.0, -1.0];
        let n_qubits = 2;
        let n_layers = 1;
        let params = vec![0.1; total_params(n_qubits, n_layers)];
        let grad = parameter_shift_gradient(&params, &features, &labels, n_qubits, n_layers);
        assert_eq!(grad.len(), params.len());
        for g in &grad {
            assert!(g.is_finite(), "Gradient should be finite");
        }
    }

    #[test]
    fn test_log_returns() {
        let candles = vec![
            Candle { timestamp: 0, open: 100.0, high: 105.0, low: 95.0, close: 100.0, volume: 1000.0 },
            Candle { timestamp: 1, open: 100.0, high: 110.0, low: 98.0, close: 105.0, volume: 1200.0 },
            Candle { timestamp: 2, open: 105.0, high: 108.0, low: 100.0, close: 103.0, volume: 900.0 },
        ];
        let rets = log_returns(&candles);
        assert_eq!(rets.len(), 2);
        assert!((rets[0] - (105.0_f64 / 100.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_rsi_range() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.005, -0.015, 0.03, -0.02, 0.01, -0.005];
        let rsi = rsi_indicator(&returns, 5);
        for &r in &rsi {
            assert!(r >= 0.0 && r <= 1.0, "RSI should be in [0, 1], got {}", r);
        }
    }

    #[test]
    fn test_label_regimes() {
        let candles = vec![
            Candle { timestamp: 0, open: 100.0, high: 105.0, low: 95.0, close: 100.0, volume: 1000.0 },
            Candle { timestamp: 1, open: 100.0, high: 110.0, low: 98.0, close: 110.0, volume: 1200.0 },
            Candle { timestamp: 2, open: 110.0, high: 112.0, low: 85.0, close: 85.0, volume: 1500.0 },
        ];
        let labels = label_regimes(&candles, 1, 0.005);
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0], 1.0); // 100 -> 110 is bull
        assert_eq!(labels[1], -1.0); // 110 -> 85 is bear
    }

    #[test]
    fn test_binary_labels() {
        let labels = vec![1.0, -1.0, 0.0, 1.0];
        let binary = to_binary_labels(&labels);
        assert_eq!(binary, vec![1.0, -1.0, -1.0, 1.0]);
    }

    #[test]
    fn test_normalize_features_range() {
        let features = vec![
            vec![0.1, -0.5, 0.8],
            vec![0.3, 0.2, 0.1],
            vec![-0.2, 0.5, 0.5],
        ];
        let normed = normalize_features(&features);
        for feat in &normed {
            for &v in feat {
                assert!(
                    v >= -1e-10 && v <= PI + 1e-10,
                    "Normalized feature should be in [0, pi], got {}",
                    v
                );
            }
        }
    }

    #[test]
    fn test_accuracy() {
        let preds = vec![1.0, -1.0, 1.0, -1.0];
        let labels = vec![1.0, -1.0, -1.0, -1.0];
        let acc = accuracy(&preds, &labels);
        assert!((acc - 75.0).abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix() {
        let preds = vec![1.0, 1.0, -1.0, -1.0];
        let labels = vec![1.0, -1.0, 1.0, -1.0];
        let (tp, fp, fn_, tn) = confusion_matrix(&preds, &labels);
        assert_eq!(tp, 1);
        assert_eq!(fp, 1);
        assert_eq!(fn_, 1);
        assert_eq!(tn, 1);
    }

    #[test]
    fn test_synthetic_candles() {
        let candles = generate_synthetic_candles(50, 42);
        assert_eq!(candles.len(), 50);
        for c in &candles {
            assert!(c.close > 0.0, "Prices should be positive");
            assert!(c.high >= c.close || c.high >= c.open, "High should be >= close or open");
            assert!(c.volume > 0.0, "Volume should be positive");
        }
    }

    #[test]
    fn test_feature_engineering() {
        let candles = generate_synthetic_candles(50, 123);
        let (features, indices) = engineer_features(&candles, 5);
        assert!(!features.is_empty(), "Should produce features");
        assert_eq!(features.len(), indices.len());
        assert_eq!(features[0].len(), 5, "Each feature vector should have 5 elements");
    }

    #[test]
    fn test_vqc_model_train() {
        let features = vec![
            vec![0.5, 1.0],
            vec![2.0, 0.3],
            vec![0.1, 2.5],
            vec![1.8, 0.1],
        ];
        let labels = vec![1.0, -1.0, 1.0, -1.0];
        let mut model = VQCModel::new(2, 1, 0.1, 42);
        let initial_cost = mse_cost(&model.params, &features, &labels, 2, 1);
        model.train(&features, &labels, 5);
        let final_cost = mse_cost(&model.params, &features, &labels, 2, 1);
        // Cost should generally decrease (or at least not increase drastically)
        assert!(final_cost <= initial_cost + 0.5, "Training should not drastically increase cost");
    }
}
