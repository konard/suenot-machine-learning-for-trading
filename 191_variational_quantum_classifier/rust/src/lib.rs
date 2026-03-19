//! Variational Quantum Classifier (VQC) for Trading
//!
//! A complete implementation of a variational quantum classifier using
//! statevector simulation, angle encoding, and parameter shift rule
//! gradient computation. Includes Bybit API integration for real market data.

use rand::Rng;
use serde::Deserialize;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────
// Complex number type
// ─────────────────────────────────────────────────────────────

/// Minimal complex number for statevector simulation.
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

    pub fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
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
}

// ─────────────────────────────────────────────────────────────
// Statevector simulator
// ─────────────────────────────────────────────────────────────

/// Quantum statevector simulator supporting RY, RZ, and CNOT gates.
pub struct Statevector {
    pub num_qubits: usize,
    pub amplitudes: Vec<Complex>,
}

impl Statevector {
    /// Create a new statevector initialized to |00...0>.
    pub fn new(num_qubits: usize) -> Self {
        let size = 1 << num_qubits;
        let mut amplitudes = vec![Complex::zero(); size];
        amplitudes[0] = Complex::one();
        Self {
            num_qubits,
            amplitudes,
        }
    }

    /// Apply a single-qubit gate (2x2 matrix) to the given qubit.
    fn apply_single_qubit_gate(&mut self, qubit: usize, gate: [[Complex; 2]; 2]) {
        let n = self.amplitudes.len();
        let step = 1 << qubit;
        let mut i = 0;
        while i < n {
            for j in i..i + step {
                let a = self.amplitudes[j];
                let b = self.amplitudes[j + step];
                self.amplitudes[j] = gate[0][0].mul(&a).add(&gate[0][1].mul(&b));
                self.amplitudes[j + step] = gate[1][0].mul(&a).add(&gate[1][1].mul(&b));
            }
            i += step << 1;
        }
    }

    /// Apply RY(theta) gate to the given qubit.
    /// RY(t) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
    pub fn apply_ry(&mut self, qubit: usize, theta: f64) {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let gate = [
            [Complex::new(c, 0.0), Complex::new(-s, 0.0)],
            [Complex::new(s, 0.0), Complex::new(c, 0.0)],
        ];
        self.apply_single_qubit_gate(qubit, gate);
    }

    /// Apply RZ(theta) gate to the given qubit.
    /// RZ(t) = [[exp(-it/2), 0], [0, exp(it/2)]]
    pub fn apply_rz(&mut self, qubit: usize, theta: f64) {
        let gate = [
            [
                Complex::new((theta / 2.0).cos(), -(theta / 2.0).sin()),
                Complex::zero(),
            ],
            [
                Complex::zero(),
                Complex::new((theta / 2.0).cos(), (theta / 2.0).sin()),
            ],
        ];
        self.apply_single_qubit_gate(qubit, gate);
    }

    /// Apply CNOT gate with the given control and target qubits.
    pub fn apply_cnot(&mut self, control: usize, target: usize) {
        let n = self.amplitudes.len();
        for i in 0..n {
            let control_bit = (i >> control) & 1;
            let target_bit = (i >> target) & 1;
            if control_bit == 1 && target_bit == 0 {
                let j = i | (1 << target);
                let tmp = self.amplitudes[i];
                self.amplitudes[i] = self.amplitudes[j];
                self.amplitudes[j] = tmp;
            }
        }
    }

    /// Get the probability of the first qubit being |1>.
    /// This sums |amplitude|^2 for all basis states where qubit 0 is |1>.
    pub fn prob_first_qubit_one(&self) -> f64 {
        let n = self.amplitudes.len();
        let mut prob = 0.0;
        for i in 0..n {
            if (i & 1) == 1 {
                prob += self.amplitudes[i].norm_sq();
            }
        }
        prob
    }
}

// ─────────────────────────────────────────────────────────────
// Variational Quantum Classifier
// ─────────────────────────────────────────────────────────────

/// Variational Quantum Classifier with angle encoding, RY/RZ rotations,
/// and CNOT entangling layers.
pub struct VQC {
    pub num_qubits: usize,
    pub num_layers: usize,
}

impl VQC {
    /// Create a new VQC.
    ///
    /// * `num_qubits` - number of qubits (= number of input features)
    /// * `num_layers` - number of variational layers
    pub fn new(num_qubits: usize, num_layers: usize) -> Self {
        Self {
            num_qubits,
            num_layers,
        }
    }

    /// Total number of trainable parameters.
    /// Each layer has 2 * num_qubits parameters (RY + RZ per qubit).
    pub fn num_params(&self) -> usize {
        2 * self.num_qubits * self.num_layers
    }

    /// Initialize random parameters in [-pi, pi].
    pub fn init_params(&self) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        (0..self.num_params())
            .map(|_| rng.gen_range(-PI..PI))
            .collect()
    }

    /// Forward pass: encode features, apply variational circuit, return P(class=1).
    ///
    /// * `features` - input features (length = num_qubits), should be scaled to [-pi, pi]
    /// * `params` - variational parameters (length = num_params())
    pub fn forward(&self, features: &[f64], params: &[f64]) -> f64 {
        assert_eq!(features.len(), self.num_qubits);
        assert_eq!(params.len(), self.num_params());

        let mut sv = Statevector::new(self.num_qubits);

        // Angle encoding: RY(feature_i) on qubit i
        for i in 0..self.num_qubits {
            sv.apply_ry(i, features[i]);
        }

        // Variational layers
        let params_per_layer = 2 * self.num_qubits;
        for l in 0..self.num_layers {
            let offset = l * params_per_layer;

            // RY and RZ rotations
            for i in 0..self.num_qubits {
                sv.apply_ry(i, params[offset + 2 * i]);
                sv.apply_rz(i, params[offset + 2 * i + 1]);
            }

            // CNOT entangling layer (linear connectivity)
            if self.num_qubits > 1 {
                for i in 0..self.num_qubits - 1 {
                    sv.apply_cnot(i, i + 1);
                }
            }
        }

        // Measurement: probability of first qubit being |1>
        sv.prob_first_qubit_one()
    }

    /// Compute gradients using the parameter shift rule combined with
    /// the chain rule through the cross-entropy loss.
    ///
    /// For each parameter theta_k, we use the parameter shift rule to get
    /// dp/dtheta_k for each sample, then multiply by dL/dp (the loss gradient
    /// w.r.t. the predicted probability) and average over the batch.
    pub fn compute_gradients(
        &self,
        features_batch: &[Vec<f64>],
        labels: &[f64],
        params: &[f64],
    ) -> Vec<f64> {
        let n_params = self.num_params();
        let n_samples = features_batch.len() as f64;
        let mut gradients = vec![0.0; n_params];
        let shift = PI / 2.0;
        let eps = 1e-7;

        for k in 0..n_params {
            let mut params_plus = params.to_vec();
            let mut params_minus = params.to_vec();
            params_plus[k] += shift;
            params_minus[k] -= shift;

            let mut grad_k = 0.0;
            for (features, &label) in features_batch.iter().zip(labels.iter()) {
                // Parameter shift rule: dp/dtheta_k
                let p_plus = self.forward(features, &params_plus);
                let p_minus = self.forward(features, &params_minus);
                let dp_dtheta = (p_plus - p_minus) / 2.0;

                // Current prediction
                let p = self.forward(features, params).clamp(eps, 1.0 - eps);

                // Chain rule: dL/dp for binary cross-entropy
                // L = -(y * ln(p) + (1-y) * ln(1-p))
                // dL/dp = -y/p + (1-y)/(1-p)
                let dl_dp = -label / p + (1.0 - label) / (1.0 - p);

                grad_k += dl_dp * dp_dtheta;
            }

            gradients[k] = grad_k / n_samples;
        }

        gradients
    }

    /// Compute binary cross-entropy loss for a batch.
    pub fn batch_loss(
        &self,
        features_batch: &[Vec<f64>],
        labels: &[f64],
        params: &[f64],
    ) -> f64 {
        let n = features_batch.len() as f64;
        let eps = 1e-7;
        let mut total_loss = 0.0;

        for (features, &label) in features_batch.iter().zip(labels.iter()) {
            let p1 = self.forward(features, params).clamp(eps, 1.0 - eps);
            let loss = -(label * p1.ln() + (1.0 - label) * (1.0 - p1).ln());
            total_loss += loss;
        }

        total_loss / n
    }

    /// Predict class probabilities for a batch of samples.
    pub fn predict_proba(&self, features_batch: &[Vec<f64>], params: &[f64]) -> Vec<f64> {
        features_batch
            .iter()
            .map(|f| self.forward(f, params))
            .collect()
    }

    /// Predict class labels (0 or 1) for a batch.
    pub fn predict(&self, features_batch: &[Vec<f64>], params: &[f64]) -> Vec<u8> {
        self.predict_proba(features_batch, params)
            .iter()
            .map(|&p| if p >= 0.5 { 1 } else { 0 })
            .collect()
    }

    /// Train the VQC using gradient descent with parameter shift rule.
    ///
    /// Returns the final parameters and a history of loss values per epoch.
    pub fn train(
        &self,
        features_batch: &[Vec<f64>],
        labels: &[f64],
        initial_params: &[f64],
        learning_rate: f64,
        num_epochs: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut params = initial_params.to_vec();
        let mut loss_history = Vec::with_capacity(num_epochs);

        for epoch in 0..num_epochs {
            let loss = self.batch_loss(features_batch, labels, &params);
            loss_history.push(loss);

            if epoch % 10 == 0 {
                println!("Epoch {}: loss = {:.6}", epoch, loss);
            }

            let gradients = self.compute_gradients(features_batch, labels, &params);

            for i in 0..params.len() {
                params[i] -= learning_rate * gradients[i];
            }
        }

        (params, loss_history)
    }

    /// Compute classification accuracy.
    pub fn accuracy(
        &self,
        features_batch: &[Vec<f64>],
        labels: &[f64],
        params: &[f64],
    ) -> f64 {
        let predictions = self.predict(features_batch, params);
        let correct: usize = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(&pred, &label)| pred == (label as u8))
            .count();
        correct as f64 / labels.len() as f64
    }
}

// ─────────────────────────────────────────────────────────────
// Bybit API integration
// ─────────────────────────────────────────────────────────────

/// Raw kline (candlestick) data from Bybit.
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
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Fetch kline data from Bybit v5 API.
///
/// * `symbol` - trading pair, e.g. "BTCUSDT"
/// * `interval` - candle interval, e.g. "15" for 15 minutes
/// * `limit` - number of candles to fetch (max 1000)
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> anyhow::Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::Client::new();
    let resp: BybitResponse = client.get(&url).send().await?.json().await?;

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

    // Bybit returns newest first, reverse to chronological order
    candles.reverse();
    Ok(candles)
}

/// Synchronous version of fetch_bybit_klines using blocking reqwest.
pub fn fetch_bybit_klines_blocking(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> anyhow::Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
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

    candles.reverse();
    Ok(candles)
}

// ─────────────────────────────────────────────────────────────
// Feature engineering
// ─────────────────────────────────────────────────────────────

/// Engineer trading features from candle data.
///
/// Features per sample (4 features):
/// 1. 1-bar log return
/// 2. 5-bar log return
/// 3. 10-bar rolling volatility
/// 4. Volume change ratio
///
/// Labels: 1.0 if next candle closes higher, 0.0 otherwise.
///
/// Returns (features, labels) where features[i] is a Vec<f64> of 4 features.
pub fn engineer_features(candles: &[Candle]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut features = Vec::new();
    let mut labels = Vec::new();
    let lookback = 10;

    if candles.len() < lookback + 2 {
        return (features, labels);
    }

    for i in lookback..candles.len() - 1 {
        // 1-bar return
        let ret_1 = (candles[i].close / candles[i - 1].close).ln();

        // 5-bar return
        let ret_5 = if i >= 5 {
            (candles[i].close / candles[i - 5].close).ln()
        } else {
            0.0
        };

        // 10-bar rolling volatility
        let mut returns = Vec::new();
        for j in (i - lookback + 1)..=i {
            returns.push((candles[j].close / candles[j - 1].close).ln());
        }
        let mean_ret: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 =
            returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / returns.len() as f64;
        let volatility = variance.sqrt();

        // Volume change
        let vol_change = if candles[i - 1].volume > 0.0 {
            (candles[i].volume - candles[i - 1].volume) / candles[i - 1].volume
        } else {
            0.0
        };

        features.push(vec![ret_1, ret_5, volatility, vol_change]);

        // Label: 1 if next candle closes higher
        let label = if candles[i + 1].close > candles[i].close {
            1.0
        } else {
            0.0
        };
        labels.push(label);
    }

    (features, labels)
}

/// Scale features to [-pi, pi] using min-max normalization.
pub fn scale_features(features: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if features.is_empty() {
        return Vec::new();
    }

    let num_features = features[0].len();
    let mut mins = vec![f64::MAX; num_features];
    let mut maxs = vec![f64::MIN; num_features];

    for sample in features {
        for (j, &val) in sample.iter().enumerate() {
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
        .map(|sample| {
            sample
                .iter()
                .enumerate()
                .map(|(j, &val)| {
                    let range = maxs[j] - mins[j];
                    if range.abs() < 1e-10 {
                        0.0
                    } else {
                        (val - mins[j]) / range * 2.0 * PI - PI
                    }
                })
                .collect()
        })
        .collect()
}

/// Split data into training and test sets.
pub fn train_test_split(
    features: &[Vec<f64>],
    labels: &[f64],
    train_ratio: f64,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let split_idx = (features.len() as f64 * train_ratio) as usize;
    let train_features = features[..split_idx].to_vec();
    let train_labels = labels[..split_idx].to_vec();
    let test_features = features[split_idx..].to_vec();
    let test_labels = labels[split_idx..].to_vec();
    (train_features, train_labels, test_features, test_labels)
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statevector_initialization() {
        let sv = Statevector::new(2);
        assert_eq!(sv.amplitudes.len(), 4);
        assert!((sv.amplitudes[0].re - 1.0).abs() < 1e-10);
        assert!((sv.amplitudes[1].re).abs() < 1e-10);
        assert!((sv.amplitudes[2].re).abs() < 1e-10);
        assert!((sv.amplitudes[3].re).abs() < 1e-10);
    }

    #[test]
    fn test_ry_gate_pi() {
        // RY(pi) on |0> should give |1>
        let mut sv = Statevector::new(1);
        sv.apply_ry(0, PI);
        // After RY(pi): |0> -> |1>
        assert!(sv.amplitudes[0].norm_sq() < 1e-10);
        assert!((sv.amplitudes[1].norm_sq() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ry_gate_half_pi() {
        // RY(pi/2) on |0> should give equal superposition
        let mut sv = Statevector::new(1);
        sv.apply_ry(0, PI / 2.0);
        assert!((sv.amplitudes[0].norm_sq() - 0.5).abs() < 1e-10);
        assert!((sv.amplitudes[1].norm_sq() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_rz_gate_preserves_probabilities() {
        // RZ only adds phase, should not change measurement probabilities from |0>
        let mut sv = Statevector::new(1);
        sv.apply_rz(0, PI / 3.0);
        assert!((sv.amplitudes[0].norm_sq() - 1.0).abs() < 1e-10);
        assert!(sv.amplitudes[1].norm_sq() < 1e-10);
    }

    #[test]
    fn test_cnot_gate() {
        // CNOT with control=0, target=1 on |10> should give |11>
        let mut sv = Statevector::new(2);
        // Create |1> on qubit 0 -> state is |01> in little-endian = index 1
        sv.apply_ry(0, PI);
        // Now apply CNOT(0, 1): since qubit 0 is |1>, qubit 1 flips
        sv.apply_cnot(0, 1);
        // Should be |11> = index 3
        assert!((sv.amplitudes[3].norm_sq() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cnot_no_flip() {
        // CNOT with control=0, target=1 on |00> should give |00>
        let mut sv = Statevector::new(2);
        sv.apply_cnot(0, 1);
        assert!((sv.amplitudes[0].norm_sq() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_statevector_normalization() {
        // After arbitrary operations, statevector should remain normalized
        let mut sv = Statevector::new(3);
        sv.apply_ry(0, 1.23);
        sv.apply_rz(1, 0.45);
        sv.apply_ry(2, 2.67);
        sv.apply_cnot(0, 1);
        sv.apply_cnot(1, 2);
        let total: f64 = sv.amplitudes.iter().map(|a| a.norm_sq()).sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vqc_forward_output_range() {
        let vqc = VQC::new(3, 2);
        let features = vec![0.5, -0.3, 1.2];
        let params = vqc.init_params();
        let prob = vqc.forward(&features, &params);
        assert!(prob >= 0.0 && prob <= 1.0);
    }

    #[test]
    fn test_vqc_num_params() {
        let vqc = VQC::new(4, 3);
        assert_eq!(vqc.num_params(), 24); // 2 * 4 * 3
    }

    #[test]
    fn test_vqc_deterministic() {
        let vqc = VQC::new(3, 2);
        let features = vec![0.5, -0.3, 1.2];
        let params = vec![0.1; vqc.num_params()];
        let p1 = vqc.forward(&features, &params);
        let p2 = vqc.forward(&features, &params);
        assert!((p1 - p2).abs() < 1e-15);
    }

    #[test]
    fn test_batch_loss_range() {
        let vqc = VQC::new(2, 1);
        let features = vec![vec![0.5, -0.3], vec![1.0, 0.2], vec![-0.5, 0.8]];
        let labels = vec![1.0, 0.0, 1.0];
        let params = vec![0.1; vqc.num_params()];
        let loss = vqc.batch_loss(&features, &labels, &params);
        assert!(loss > 0.0); // Cross-entropy is always positive
    }

    #[test]
    fn test_gradient_finite_difference() {
        // Verify parameter shift rule gradients match finite difference
        let vqc = VQC::new(2, 1);
        let features = vec![vec![0.5, -0.3], vec![1.0, 0.2]];
        let labels = vec![1.0, 0.0];
        let params = vec![0.3, -0.5, 0.7, 1.1];
        let gradients = vqc.compute_gradients(&features, &labels, &params);

        let eps = 1e-5;
        for k in 0..params.len() {
            let mut p_plus = params.clone();
            let mut p_minus = params.clone();
            p_plus[k] += eps;
            p_minus[k] -= eps;
            let fd_grad =
                (vqc.batch_loss(&features, &labels, &p_plus)
                    - vqc.batch_loss(&features, &labels, &p_minus))
                    / (2.0 * eps);
            assert!(
                (gradients[k] - fd_grad).abs() < 0.05,
                "Gradient mismatch at param {}: shift={}, fd={}",
                k,
                gradients[k],
                fd_grad
            );
        }
    }

    #[test]
    fn test_scale_features() {
        let features = vec![vec![1.0, 10.0], vec![3.0, 20.0], vec![5.0, 30.0]];
        let scaled = scale_features(&features);
        // Min should map to -pi, max to pi
        assert!((scaled[0][0] - (-PI)).abs() < 1e-10);
        assert!((scaled[2][0] - PI).abs() < 1e-10);
        assert!((scaled[1][0]).abs() < 1e-10); // midpoint -> 0
    }

    #[test]
    fn test_train_test_split() {
        let features = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ];
        let labels = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let (train_f, train_l, test_f, test_l) =
            train_test_split(&features, &labels, 0.6);
        assert_eq!(train_f.len(), 3);
        assert_eq!(train_l.len(), 3);
        assert_eq!(test_f.len(), 2);
        assert_eq!(test_l.len(), 2);
    }

    #[test]
    fn test_predict() {
        let vqc = VQC::new(2, 1);
        let features = vec![vec![0.5, -0.3], vec![1.0, 0.2]];
        let params = vec![0.1; vqc.num_params()];
        let preds = vqc.predict(&features, &params);
        assert_eq!(preds.len(), 2);
        for &p in &preds {
            assert!(p == 0 || p == 1);
        }
    }

    #[test]
    fn test_engineer_features_with_synthetic_data() {
        // Create synthetic candles
        let candles: Vec<Candle> = (0..25)
            .map(|i| Candle {
                timestamp: i as u64 * 60000,
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0 + i as f64 * 10.0,
            })
            .collect();

        let (features, labels) = engineer_features(&candles);
        assert!(!features.is_empty());
        assert_eq!(features.len(), labels.len());
        assert_eq!(features[0].len(), 4);
    }

    #[test]
    fn test_vqc_training_reduces_loss() {
        let vqc = VQC::new(2, 1);

        // Simple dataset: features near [1, 1] -> class 1, near [-1, -1] -> class 0
        let features = vec![
            vec![0.8, 0.9],
            vec![0.7, 0.8],
            vec![-0.8, -0.9],
            vec![-0.7, -0.8],
        ];
        let labels = vec![1.0, 1.0, 0.0, 0.0];

        let params = vec![0.1; vqc.num_params()];
        let initial_loss = vqc.batch_loss(&features, &labels, &params);

        let (final_params, _loss_history) = vqc.train(&features, &labels, &params, 0.5, 30);
        let final_loss = vqc.batch_loss(&features, &labels, &final_params);

        assert!(
            final_loss < initial_loss,
            "Training should reduce loss: initial={}, final={}",
            initial_loss,
            final_loss
        );
    }

    #[test]
    fn test_complex_arithmetic() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let c = a.mul(&b);
        // (1+2i)(3+4i) = 3+4i+6i+8i^2 = -5+10i
        assert!((c.re - (-5.0)).abs() < 1e-10);
        assert!((c.im - 10.0).abs() < 1e-10);

        let d = a.add(&b);
        assert!((d.re - 4.0).abs() < 1e-10);
        assert!((d.im - 6.0).abs() < 1e-10);
    }
}
