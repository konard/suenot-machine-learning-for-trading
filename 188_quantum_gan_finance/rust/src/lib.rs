//! Quantum GAN for Finance
//!
//! Classical simulation of a quantum generative adversarial network
//! for synthetic financial data generation.

use rand::Rng;
use serde::Deserialize;
use std::f64::consts::PI;

// ─── Complex number type ───────────────────────────────────────────

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

// ─── Quantum Generator ────────────────────────────────────────────

/// Simulated quantum circuit generator with parameterized rotations
/// and entanglement layers.
pub struct QuantumGenerator {
    pub n_qubits: usize,
    pub n_layers: usize,
    /// Parameters: n_layers * n_qubits * 3 rotation angles (Rx, Ry, Rz per qubit per layer)
    pub params: Vec<f64>,
}

impl QuantumGenerator {
    /// Create a new quantum generator with random parameters.
    pub fn new(n_qubits: usize, n_layers: usize) -> Self {
        let mut rng = rand::thread_rng();
        let n_params = n_layers * n_qubits * 3;
        let params: Vec<f64> = (0..n_params).map(|_| rng.gen::<f64>() * 2.0 * PI).collect();
        Self {
            n_qubits,
            n_layers,
            params,
        }
    }

    /// Create with specific parameters (for testing / gradient computation).
    pub fn with_params(n_qubits: usize, n_layers: usize, params: Vec<f64>) -> Self {
        Self {
            n_qubits,
            n_layers,
            params,
        }
    }

    /// Total number of trainable parameters.
    pub fn num_params(&self) -> usize {
        self.params.len()
    }

    /// Simulate the quantum circuit and return the probability distribution
    /// over 2^n_qubits basis states.
    pub fn forward(&self) -> Vec<f64> {
        let dim = 1 << self.n_qubits;
        // Initialize state |00...0>
        let mut state = vec![Complex::zero(); dim];
        state[0] = Complex::new(1.0, 0.0);

        let mut param_idx = 0;
        for _layer in 0..self.n_layers {
            // Apply rotation gates to each qubit
            for qubit in 0..self.n_qubits {
                let theta_x = self.params[param_idx];
                let theta_y = self.params[param_idx + 1];
                let theta_z = self.params[param_idx + 2];
                param_idx += 3;

                state = apply_rx(&state, qubit, self.n_qubits, theta_x);
                state = apply_ry(&state, qubit, self.n_qubits, theta_y);
                state = apply_rz(&state, qubit, self.n_qubits, theta_z);
            }

            // Apply CNOT entangling gates between adjacent qubits
            for qubit in 0..self.n_qubits.saturating_sub(1) {
                state = apply_cnot(&state, qubit, qubit + 1, self.n_qubits);
            }
        }

        // Compute probability distribution from amplitudes
        state.iter().map(|c| c.norm_sq()).collect()
    }

    /// Sample from the output distribution and map to continuous values in [0, 1].
    pub fn generate_samples(&self, n_samples: usize) -> Vec<f64> {
        let probs = self.forward();
        let dim = probs.len();
        let mut rng = rand::thread_rng();
        let mut samples = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            // Sample a basis state index from the probability distribution
            let r: f64 = rng.gen();
            let mut cumulative = 0.0;
            let mut sampled_idx = dim - 1;
            for (i, &p) in probs.iter().enumerate() {
                cumulative += p;
                if r < cumulative {
                    sampled_idx = i;
                    break;
                }
            }
            // Map index to [0, 1]
            let value = (sampled_idx as f64 + rng.gen::<f64>()) / dim as f64;
            samples.push(value);
        }

        samples
    }
}

// ─── Quantum gate operations ──────────────────────────────────────

/// Apply Rx(theta) rotation gate to a specific qubit in the state vector.
fn apply_rx(state: &[Complex], qubit: usize, n_qubits: usize, theta: f64) -> Vec<Complex> {
    let cos = (theta / 2.0).cos();
    let sin = (theta / 2.0).sin();
    // Rx = [[cos, -i*sin], [-i*sin, cos]]
    let m00 = Complex::new(cos, 0.0);
    let m01 = Complex::new(0.0, -sin);
    let m10 = Complex::new(0.0, -sin);
    let m11 = Complex::new(cos, 0.0);
    apply_single_qubit_gate(state, qubit, n_qubits, m00, m01, m10, m11)
}

/// Apply Ry(theta) rotation gate to a specific qubit in the state vector.
fn apply_ry(state: &[Complex], qubit: usize, n_qubits: usize, theta: f64) -> Vec<Complex> {
    let cos = (theta / 2.0).cos();
    let sin = (theta / 2.0).sin();
    // Ry = [[cos, -sin], [sin, cos]]
    let m00 = Complex::new(cos, 0.0);
    let m01 = Complex::new(-sin, 0.0);
    let m10 = Complex::new(sin, 0.0);
    let m11 = Complex::new(cos, 0.0);
    apply_single_qubit_gate(state, qubit, n_qubits, m00, m01, m10, m11)
}

/// Apply Rz(theta) rotation gate to a specific qubit in the state vector.
fn apply_rz(state: &[Complex], qubit: usize, n_qubits: usize, theta: f64) -> Vec<Complex> {
    let cos = (theta / 2.0).cos();
    let sin = (theta / 2.0).sin();
    // Rz = [[e^{-i*theta/2}, 0], [0, e^{i*theta/2}]]
    let m00 = Complex::new(cos, -sin);
    let m01 = Complex::zero();
    let m10 = Complex::zero();
    let m11 = Complex::new(cos, sin);
    apply_single_qubit_gate(state, qubit, n_qubits, m00, m01, m10, m11)
}

/// Apply a single-qubit gate (given as 2x2 matrix) to a specific qubit.
fn apply_single_qubit_gate(
    state: &[Complex],
    qubit: usize,
    n_qubits: usize,
    m00: Complex,
    m01: Complex,
    m10: Complex,
    m11: Complex,
) -> Vec<Complex> {
    let dim = 1 << n_qubits;
    let mut new_state = vec![Complex::zero(); dim];
    let bit = 1 << (n_qubits - 1 - qubit);

    for i in 0..dim {
        if i & bit == 0 {
            let j = i | bit;
            // |0> component: m00 * state[i] + m01 * state[j]
            new_state[i] = m00.mul(&state[i]).add(&m01.mul(&state[j]));
            // |1> component: m10 * state[i] + m11 * state[j]
            new_state[j] = m10.mul(&state[i]).add(&m11.mul(&state[j]));
        }
    }

    new_state
}

/// Apply CNOT gate with control and target qubits.
fn apply_cnot(
    state: &[Complex],
    control: usize,
    target: usize,
    n_qubits: usize,
) -> Vec<Complex> {
    let dim = 1 << n_qubits;
    let mut new_state = state.to_vec();
    let control_bit = 1 << (n_qubits - 1 - control);
    let target_bit = 1 << (n_qubits - 1 - target);

    for i in 0..dim {
        // If control qubit is |1>, flip the target qubit
        if (i & control_bit) != 0 && (i & target_bit) == 0 {
            let j = i | target_bit;
            let tmp = new_state[i];
            new_state[i] = new_state[j];
            new_state[j] = tmp;
        }
    }

    new_state
}

// ─── Classical Discriminator ──────────────────────────────────────

/// A simple feedforward neural network discriminator.
pub struct ClassicalDiscriminator {
    /// Weights and biases for each layer.
    pub layers: Vec<DenseLayer>,
}

/// A dense (fully-connected) layer.
pub struct DenseLayer {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub input_size: usize,
    pub output_size: usize,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / input_size as f64).sqrt();
        let weights = (0..output_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| rng.gen::<f64>() * scale - scale / 2.0)
                    .collect()
            })
            .collect();
        let biases = vec![0.0; output_size];
        Self {
            weights,
            biases,
            input_size,
            output_size,
        }
    }

    /// Forward pass through this layer (without activation).
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; self.output_size];
        for i in 0..self.output_size {
            let mut sum = self.biases[i];
            for j in 0..self.input_size {
                sum += self.weights[i][j] * input[j];
            }
            output[i] = sum;
        }
        output
    }
}

impl ClassicalDiscriminator {
    /// Create a discriminator with given layer sizes.
    /// E.g., layer_sizes = [1, 16, 16, 1] for a 3-layer network.
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(DenseLayer::new(layer_sizes[i], layer_sizes[i + 1]));
        }
        Self { layers }
    }

    /// Forward pass through the discriminator.
    /// Returns a value in (0, 1) representing P(real).
    pub fn forward(&self, input: &[f64]) -> f64 {
        let mut x = input.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x);
            if i < self.layers.len() - 1 {
                // ReLU activation for hidden layers
                x.iter_mut().for_each(|v| *v = v.max(0.0));
            }
        }
        // Sigmoid activation for output
        sigmoid(x[0])
    }

    /// Update weights using simple gradient descent for a single sample.
    /// `input` is the discriminator input, `target` is the desired output (1 for real, 0 for fake).
    pub fn train_step(&mut self, input: &[f64], target: f64, lr: f64) {
        // Forward pass storing intermediate activations
        let mut activations = vec![input.to_vec()];
        let mut x = input.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x);
            if i < self.layers.len() - 1 {
                x.iter_mut().for_each(|v| *v = v.max(0.0));
            }
            activations.push(x.clone());
        }

        let output = sigmoid(x[0]);
        // d(BCE)/d(output) = output - target (for sigmoid + binary cross entropy)
        let mut delta = vec![output - target];

        // Backpropagation
        for i in (0..self.layers.len()).rev() {
            let prev_act = &activations[i];
            let curr_delta = delta.clone();

            // Update weights and biases
            for j in 0..self.layers[i].output_size {
                self.layers[i].biases[j] -= lr * curr_delta[j];
                for k in 0..self.layers[i].input_size {
                    self.layers[i].weights[j][k] -= lr * curr_delta[j] * prev_act[k];
                }
            }

            // Compute delta for previous layer
            if i > 0 {
                let mut new_delta = vec![0.0; self.layers[i].input_size];
                for j in 0..self.layers[i].output_size {
                    for k in 0..self.layers[i].input_size {
                        new_delta[k] += self.layers[i].weights[j][k] * curr_delta[j];
                    }
                }
                // Apply ReLU derivative
                let prev = &activations[i];
                for (k, d) in new_delta.iter_mut().enumerate() {
                    if prev[k] <= 0.0 {
                        *d = 0.0;
                    }
                }
                delta = new_delta;
            }
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ─── QGAN Trainer ─────────────────────────────────────────────────

/// Quantum GAN trainer that coordinates generator and discriminator training.
pub struct QGANTrainer {
    pub generator: QuantumGenerator,
    pub discriminator: ClassicalDiscriminator,
    pub gen_lr: f64,
    pub disc_lr: f64,
}

impl QGANTrainer {
    pub fn new(n_qubits: usize, n_layers: usize, gen_lr: f64, disc_lr: f64) -> Self {
        let generator = QuantumGenerator::new(n_qubits, n_layers);
        let discriminator = ClassicalDiscriminator::new(&[1, 16, 16, 1]);
        Self {
            generator,
            discriminator,
            gen_lr,
            disc_lr,
        }
    }

    /// Train the QGAN for a given number of epochs.
    /// `real_data` should be normalized to [0, 1].
    pub fn train(&mut self, real_data: &[f64], epochs: usize, batch_size: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut losses = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            // ── Train discriminator ──
            let fake_samples = self.generator.generate_samples(batch_size);

            // Train on real data
            for _ in 0..batch_size.min(real_data.len()) {
                let idx = rng.gen_range(0..real_data.len());
                let input = [real_data[idx]];
                self.discriminator.train_step(&input, 1.0, self.disc_lr);
            }

            // Train on fake data
            for sample in &fake_samples {
                let input = [*sample];
                self.discriminator.train_step(&input, 0.0, self.disc_lr);
            }

            // ── Train generator (parameter shift rule) ──
            let n_params = self.generator.num_params();
            let mut gradients = vec![0.0; n_params];

            for i in 0..n_params {
                // Forward evaluation with +pi/2 shift
                let mut params_plus = self.generator.params.clone();
                params_plus[i] += PI / 2.0;
                let gen_plus =
                    QuantumGenerator::with_params(self.generator.n_qubits, self.generator.n_layers, params_plus);
                let samples_plus = gen_plus.generate_samples(batch_size);
                let loss_plus: f64 = samples_plus
                    .iter()
                    .map(|s| {
                        let d = self.discriminator.forward(&[*s]);
                        -(d + 1e-10).ln()
                    })
                    .sum::<f64>()
                    / batch_size as f64;

                // Forward evaluation with -pi/2 shift
                let mut params_minus = self.generator.params.clone();
                params_minus[i] -= PI / 2.0;
                let gen_minus = QuantumGenerator::with_params(
                    self.generator.n_qubits,
                    self.generator.n_layers,
                    params_minus,
                );
                let samples_minus = gen_minus.generate_samples(batch_size);
                let loss_minus: f64 = samples_minus
                    .iter()
                    .map(|s| {
                        let d = self.discriminator.forward(&[*s]);
                        -(d + 1e-10).ln()
                    })
                    .sum::<f64>()
                    / batch_size as f64;

                gradients[i] = (loss_plus - loss_minus) / 2.0;
            }

            // Update generator parameters
            for i in 0..n_params {
                self.generator.params[i] -= self.gen_lr * gradients[i];
            }

            // Compute epoch loss
            let fake_samples = self.generator.generate_samples(batch_size);
            let gen_loss: f64 = fake_samples
                .iter()
                .map(|s| {
                    let d = self.discriminator.forward(&[*s]);
                    -(d + 1e-10).ln()
                })
                .sum::<f64>()
                / batch_size as f64;

            losses.push(gen_loss);

            if epoch % 10 == 0 {
                println!("Epoch {}/{}: gen_loss = {:.4}", epoch + 1, epochs, gen_loss);
            }
        }

        losses
    }

    /// Generate synthetic samples using the trained generator.
    pub fn generate(&self, n_samples: usize) -> Vec<f64> {
        self.generator.generate_samples(n_samples)
    }
}

// ─── Financial data utilities ─────────────────────────────────────

/// Normalize data to [0, 1] range.
pub fn normalize(data: &[f64]) -> (Vec<f64>, f64, f64) {
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    if range < 1e-12 {
        return (vec![0.5; data.len()], min, max);
    }
    let normalized = data.iter().map(|x| (x - min) / range).collect();
    (normalized, min, max)
}

/// Denormalize data from [0, 1] back to original range.
pub fn denormalize(data: &[f64], min: f64, max: f64) -> Vec<f64> {
    let range = max - min;
    data.iter().map(|x| x * range + min).collect()
}

/// Compute log returns from a price series.
pub fn compute_log_returns(prices: &[f64]) -> Vec<f64> {
    prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}

/// Compute mean of a slice.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Compute standard deviation of a slice.
pub fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    let variance = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    variance.sqrt()
}

/// Compute skewness of a slice.
pub fn skewness(data: &[f64]) -> f64 {
    if data.len() < 3 {
        return 0.0;
    }
    let m = mean(data);
    let s = std_dev(data);
    if s < 1e-12 {
        return 0.0;
    }
    let n = data.len() as f64;
    let sum_cubed: f64 = data.iter().map(|x| ((x - m) / s).powi(3)).sum();
    sum_cubed / n
}

/// Compute excess kurtosis of a slice.
pub fn kurtosis(data: &[f64]) -> f64 {
    if data.len() < 4 {
        return 0.0;
    }
    let m = mean(data);
    let s = std_dev(data);
    if s < 1e-12 {
        return 0.0;
    }
    let n = data.len() as f64;
    let sum_fourth: f64 = data.iter().map(|x| ((x - m) / s).powi(4)).sum();
    sum_fourth / n - 3.0
}

// ─── Bybit API ────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitKlineResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitKlineResult {
    pub list: Vec<Vec<String>>,
}

/// Fetch kline data from Bybit API.
/// Returns a vector of closing prices.
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> anyhow::Result<Vec<f64>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitKlineResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    // Kline list format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
    // Data comes in reverse chronological order, so we reverse it
    let mut prices: Vec<f64> = resp
        .result
        .list
        .iter()
        .filter_map(|kline| {
            if kline.len() >= 5 {
                kline[4].parse::<f64>().ok()
            } else {
                None
            }
        })
        .collect();

    prices.reverse();
    Ok(prices)
}

// ─── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_operations() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);

        let product = a.mul(&b);
        assert!((product.re - (-5.0)).abs() < 1e-10);
        assert!((product.im - 10.0).abs() < 1e-10);

        let sum = a.add(&b);
        assert!((sum.re - 4.0).abs() < 1e-10);
        assert!((sum.im - 6.0).abs() < 1e-10);

        assert!((a.norm_sq() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_generator_probabilities_sum_to_one() {
        let gen = QuantumGenerator::new(3, 2);
        let probs = gen.forward();

        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-8,
            "Probabilities should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_quantum_generator_output_dimension() {
        let n_qubits = 3;
        let gen = QuantumGenerator::new(n_qubits, 2);
        let probs = gen.forward();
        assert_eq!(probs.len(), 1 << n_qubits);
    }

    #[test]
    fn test_quantum_generator_samples_in_range() {
        let gen = QuantumGenerator::new(3, 2);
        let samples = gen.generate_samples(100);

        for s in &samples {
            assert!(*s >= 0.0 && *s <= 1.0, "Sample {} out of [0,1] range", s);
        }
    }

    #[test]
    fn test_discriminator_output_range() {
        let disc = ClassicalDiscriminator::new(&[1, 8, 1]);
        for i in 0..20 {
            let val = i as f64 / 20.0;
            let output = disc.forward(&[val]);
            assert!(
                output > 0.0 && output < 1.0,
                "Discriminator output {} out of (0,1) range",
                output
            );
        }
    }

    #[test]
    fn test_normalize_denormalize_roundtrip() {
        let data = vec![-0.05, 0.02, 0.1, -0.03, 0.07];
        let (normalized, min, max) = normalize(&data);

        for v in &normalized {
            assert!(*v >= 0.0 && *v <= 1.0);
        }

        let recovered = denormalize(&normalized, min, max);
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!(
                (orig - rec).abs() < 1e-10,
                "Roundtrip failed: {} != {}",
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_log_returns() {
        let prices = vec![100.0, 105.0, 102.0, 108.0];
        let returns = compute_log_returns(&prices);
        assert_eq!(returns.len(), 3);
        assert!((returns[0] - (105.0_f64 / 100.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_statistical_functions() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&data) - 3.0).abs() < 1e-10);

        let s = std_dev(&data);
        // std dev of [1,2,3,4,5] with Bessel correction = sqrt(2.5) ≈ 1.5811
        assert!((s - (2.5_f64).sqrt()).abs() < 1e-4);
    }

    #[test]
    fn test_parameter_shift_gradient() {
        // Verify that the parameter shift rule produces a non-trivial gradient
        let gen = QuantumGenerator::new(2, 1);
        let n_params = gen.num_params();

        let disc = ClassicalDiscriminator::new(&[1, 8, 1]);

        let mut has_nonzero = false;
        for i in 0..n_params {
            let mut params_plus = gen.params.clone();
            params_plus[i] += PI / 2.0;
            let gen_plus = QuantumGenerator::with_params(2, 1, params_plus);
            let samples_plus = gen_plus.generate_samples(50);
            let loss_plus: f64 = samples_plus.iter().map(|s| disc.forward(&[*s])).sum::<f64>();

            let mut params_minus = gen.params.clone();
            params_minus[i] -= PI / 2.0;
            let gen_minus = QuantumGenerator::with_params(2, 1, params_minus);
            let samples_minus = gen_minus.generate_samples(50);
            let loss_minus: f64 = samples_minus.iter().map(|s| disc.forward(&[*s])).sum::<f64>();

            let grad = (loss_plus - loss_minus) / 2.0;
            if grad.abs() > 1e-6 {
                has_nonzero = true;
            }
        }

        assert!(has_nonzero, "At least one gradient should be non-zero");
    }
}
