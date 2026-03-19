use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;
use std::f64::consts::PI;

// =============================================================================
// Complex number type for quantum state simulation
// =============================================================================

#[derive(Clone, Copy, Debug)]
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

// =============================================================================
// Quantum Circuit Simulator (Generator)
// =============================================================================

/// Parameterized quantum circuit that acts as the generator in the QGAN.
/// Maintains a full 2^n complex state vector and applies rotation and
/// entangling gates to produce probability distributions via Born-rule sampling.
pub struct QuantumCircuit {
    pub num_qubits: usize,
    pub num_layers: usize,
    pub params: Vec<f64>,
    state: Vec<Complex>,
}

impl QuantumCircuit {
    /// Create a new quantum circuit with the given number of qubits and layers.
    /// Each layer has one R_y parameter per qubit, so total params = num_qubits * num_layers.
    pub fn new(num_qubits: usize, num_layers: usize) -> Self {
        let num_params = num_qubits * num_layers;
        let mut rng = rand::thread_rng();
        let params: Vec<f64> = (0..num_params)
            .map(|_| rng.gen_range(0.0..2.0 * PI))
            .collect();

        let dim = 1 << num_qubits;
        let mut state = vec![Complex::zero(); dim];
        state[0] = Complex::new(1.0, 0.0); // |0...0>

        Self {
            num_qubits,
            num_layers,
            params,
            state,
        }
    }

    /// Reset state to |0...0>.
    fn reset_state(&mut self) {
        for s in self.state.iter_mut() {
            *s = Complex::zero();
        }
        self.state[0] = Complex::new(1.0, 0.0);
    }

    /// Apply R_y(theta) gate to the specified qubit.
    /// R_y(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
    fn apply_ry(&mut self, qubit: usize, theta: f64) {
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();
        let dim = 1 << self.num_qubits;
        let mask = 1 << qubit;

        let mut i = 0;
        while i < dim {
            // Find pairs of indices differing only in the target qubit bit
            if i & mask == 0 {
                let j = i | mask;
                let a = self.state[i];
                let b = self.state[j];
                self.state[i] = Complex {
                    re: cos_half * a.re - sin_half * b.re,
                    im: cos_half * a.im - sin_half * b.im,
                };
                self.state[j] = Complex {
                    re: sin_half * a.re + cos_half * b.re,
                    im: sin_half * a.im + cos_half * b.im,
                };
            }
            // Skip to next pair
            i += 1;
        }
    }

    /// Apply CNOT gate with control and target qubits.
    /// Flips the target qubit if the control qubit is |1>.
    fn apply_cnot(&mut self, control: usize, target: usize) {
        let dim = 1 << self.num_qubits;
        let control_mask = 1 << control;
        let target_mask = 1 << target;

        for i in 0..dim {
            // Only act when control is 1 and target is 0
            if (i & control_mask != 0) && (i & target_mask == 0) {
                let j = i | target_mask;
                self.state.swap(i, j);
            }
        }
    }

    /// Run the full parameterized circuit: alternating layers of R_y rotations
    /// and CNOT entanglement in a ring topology.
    pub fn run_circuit(&mut self) {
        self.reset_state();
        for layer in 0..self.num_layers {
            // Rotation layer: R_y on each qubit
            for q in 0..self.num_qubits {
                let param_idx = layer * self.num_qubits + q;
                self.apply_ry(q, self.params[param_idx]);
            }
            // Entanglement layer: CNOT ring
            for q in 0..self.num_qubits {
                let target = (q + 1) % self.num_qubits;
                self.apply_cnot(q, target);
            }
        }
    }

    /// Get the Born-rule probability distribution: p(x) = |<x|psi>|^2.
    pub fn get_probabilities(&mut self) -> Vec<f64> {
        self.run_circuit();
        self.state.iter().map(|c| c.norm_sq()).collect()
    }

    /// Sample from the Born machine distribution.
    /// Returns `num_samples` indices drawn according to the probability distribution.
    pub fn sample(&mut self, num_samples: usize) -> Vec<usize> {
        let probs = self.get_probabilities();
        let mut rng = rand::thread_rng();
        let mut samples = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            let r: f64 = rng.gen();
            let mut cumulative = 0.0;
            let mut chosen = probs.len() - 1;
            for (idx, &p) in probs.iter().enumerate() {
                cumulative += p;
                if r < cumulative {
                    chosen = idx;
                    break;
                }
            }
            samples.push(chosen);
        }
        samples
    }

    /// Map sampled indices to real-valued returns in [r_min, r_max].
    pub fn sample_returns(&mut self, num_samples: usize, r_min: f64, r_max: f64) -> Vec<f64> {
        let indices = self.sample(num_samples);
        let num_bins = 1 << self.num_qubits;
        indices
            .iter()
            .map(|&idx| r_min + (idx as f64) * (r_max - r_min) / ((num_bins - 1) as f64))
            .collect()
    }

    /// Run circuit with a specific set of parameters (used for parameter-shift).
    pub fn get_probabilities_with_params(&mut self, params: &[f64]) -> Vec<f64> {
        let old_params = self.params.clone();
        self.params = params.to_vec();
        let probs = self.get_probabilities();
        self.params = old_params;
        probs
    }
}

// =============================================================================
// Classical Discriminator (simple neural network)
// =============================================================================

/// A simple two-layer feedforward neural network serving as the discriminator.
/// Architecture: input -> Linear(hidden) -> ReLU -> Linear(1) -> Sigmoid
pub struct Discriminator {
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
    pub learning_rate: f64,
    hidden_size: usize,
}

impl Discriminator {
    /// Create a discriminator with given input and hidden dimensions.
    pub fn new(input_size: usize, hidden_size: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let scale1 = (2.0 / input_size as f64).sqrt();
        let scale2 = (2.0 / hidden_size as f64).sqrt();

        let w1 = Array2::from_shape_fn((hidden_size, input_size), |_| {
            rng.gen_range(-scale1..scale1)
        });
        let b1 = Array1::zeros(hidden_size);
        let w2 = Array2::from_shape_fn((1, hidden_size), |_| {
            rng.gen_range(-scale2..scale2)
        });
        let b2 = Array1::zeros(1);

        Self {
            w1,
            b1,
            w2,
            b2,
            learning_rate,
            hidden_size,
        }
    }

    /// Forward pass: returns (output, hidden_pre_activation, hidden_post_activation).
    fn forward_detailed(&self, x: &Array1<f64>) -> (f64, Array1<f64>, Array1<f64>) {
        // Hidden layer: z1 = W1 * x + b1
        let z1 = self.w1.dot(x) + &self.b1;
        // ReLU activation
        let h1 = z1.mapv(|v| if v > 0.0 { v } else { 0.0 });
        // Output layer: z2 = W2 * h1 + b2
        let z2 = self.w2.dot(&h1) + &self.b2;
        // Sigmoid
        let output = sigmoid(z2[0]);
        (output, z1, h1)
    }

    /// Forward pass returning only the scalar output.
    pub fn forward(&self, x: &Array1<f64>) -> f64 {
        self.forward_detailed(x).0
    }

    /// Train the discriminator on a batch of real and fake samples.
    /// Returns the average discriminator loss.
    pub fn train_step(&mut self, real_samples: &[Array1<f64>], fake_samples: &[Array1<f64>]) -> f64 {
        let mut total_loss = 0.0;
        let batch_size = real_samples.len() + fake_samples.len();

        // Accumulate gradients
        let mut dw1 = Array2::zeros(self.w1.raw_dim());
        let mut db1 = Array1::zeros(self.hidden_size);
        let mut dw2 = Array2::zeros(self.w2.raw_dim());
        let mut db2 = Array1::zeros(1);

        // Real samples: label = 1, loss = -log(D(x))
        for x in real_samples {
            let (output, z1, h1) = self.forward_detailed(x);
            let loss = -(output + 1e-8).ln();
            total_loss += loss;

            // Gradient of -log(sigmoid(z)) = sigmoid(z) - 1 = output - 1
            let dz2 = output - 1.0;
            self.accumulate_gradients(x, &z1, &h1, dz2, &mut dw1, &mut db1, &mut dw2, &mut db2);
        }

        // Fake samples: label = 0, loss = -log(1 - D(x))
        for x in fake_samples {
            let (output, z1, h1) = self.forward_detailed(x);
            let loss = -(1.0 - output + 1e-8).ln();
            total_loss += loss;

            // Gradient of -log(1 - sigmoid(z)) = sigmoid(z) = output
            let dz2 = output;
            self.accumulate_gradients(x, &z1, &h1, dz2, &mut dw1, &mut db1, &mut dw2, &mut db2);
        }

        // Apply averaged gradients
        let scale = self.learning_rate / batch_size as f64;
        self.w1 = &self.w1 - &(dw1 * scale);
        self.b1 = &self.b1 - &(db1 * scale);
        self.w2 = &self.w2 - &(dw2 * scale);
        self.b2 = &self.b2 - &(db2 * scale);

        total_loss / batch_size as f64
    }

    fn accumulate_gradients(
        &self,
        x: &Array1<f64>,
        z1: &Array1<f64>,
        h1: &Array1<f64>,
        dz2: f64,
        dw1: &mut Array2<f64>,
        db1: &mut Array1<f64>,
        dw2: &mut Array2<f64>,
        db2: &mut Array1<f64>,
    ) {
        // dL/dW2 = dz2 * h1^T
        for j in 0..self.hidden_size {
            dw2[[0, j]] += dz2 * h1[j];
        }
        db2[0] += dz2;

        // dL/dh1 = W2^T * dz2
        // dL/dz1 = dL/dh1 * relu'(z1)
        for i in 0..self.hidden_size {
            let dh1_i = self.w2[[0, i]] * dz2;
            let dz1_i = if z1[i] > 0.0 { dh1_i } else { 0.0 };
            db1[i] += dz1_i;
            for j in 0..x.len() {
                dw1[[i, j]] += dz1_i * x[j];
            }
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// =============================================================================
// Hybrid QGAN Training
// =============================================================================

/// Hybrid Quantum GAN that combines a quantum circuit generator with a
/// classical neural network discriminator.
pub struct QuantumGAN {
    pub generator: QuantumCircuit,
    pub discriminator: Discriminator,
    pub gen_learning_rate: f64,
    pub r_min: f64,
    pub r_max: f64,
}

impl QuantumGAN {
    /// Create a new QGAN with default hyperparameters.
    pub fn new(num_qubits: usize, num_layers: usize, r_min: f64, r_max: f64) -> Self {
        let generator = QuantumCircuit::new(num_qubits, num_layers);
        let discriminator = Discriminator::new(1, 16, 0.01);

        Self {
            generator,
            discriminator,
            gen_learning_rate: 0.05,
            r_min,
            r_max,
        }
    }

    /// Discretize real return values into bin probability distribution.
    pub fn discretize_returns(&self, returns: &[f64]) -> Vec<f64> {
        let num_bins = 1 << self.generator.num_qubits;
        let mut hist = vec![0.0_f64; num_bins];
        let bin_width = (self.r_max - self.r_min) / (num_bins - 1) as f64;

        for &r in returns {
            let clamped = r.clamp(self.r_min, self.r_max);
            let bin = ((clamped - self.r_min) / bin_width).round() as usize;
            let bin = bin.min(num_bins - 1);
            hist[bin] += 1.0;
        }

        // Normalize to probability distribution
        let total: f64 = hist.iter().sum();
        if total > 0.0 {
            hist.iter_mut().for_each(|h| *h /= total);
        }
        hist
    }

    /// Map a bin index to a return value.
    fn bin_to_return(&self, bin: usize) -> f64 {
        let num_bins = 1 << self.generator.num_qubits;
        self.r_min + (bin as f64) * (self.r_max - self.r_min) / ((num_bins - 1) as f64)
    }

    /// Train the QGAN for the specified number of epochs.
    /// Returns a vector of (discriminator_loss, generator_loss) per epoch.
    pub fn train(&mut self, real_returns: &[f64], epochs: usize, batch_size: usize) -> Vec<(f64, f64)> {
        let mut losses = Vec::with_capacity(epochs);
        let mut rng = rand::thread_rng();

        for epoch in 0..epochs {
            // --- Discriminator step ---
            // Sample real data batch
            let real_batch: Vec<Array1<f64>> = (0..batch_size)
                .map(|_| {
                    let idx = rng.gen_range(0..real_returns.len());
                    Array1::from_vec(vec![real_returns[idx]])
                })
                .collect();

            // Sample fake data from generator
            let fake_returns = self.generator.sample_returns(batch_size, self.r_min, self.r_max);
            let fake_batch: Vec<Array1<f64>> = fake_returns
                .iter()
                .map(|&r| Array1::from_vec(vec![r]))
                .collect();

            let d_loss = self.discriminator.train_step(&real_batch, &fake_batch);

            // --- Generator step (parameter-shift rule) ---
            let num_params = self.generator.params.len();
            let mut g_loss = 0.0;

            for p_idx in 0..num_params {
                // Shift parameter +pi/2
                let mut params_plus = self.generator.params.clone();
                params_plus[p_idx] += PI / 2.0;
                let probs_plus = self.generator.get_probabilities_with_params(&params_plus);

                // Shift parameter -pi/2
                let mut params_minus = self.generator.params.clone();
                params_minus[p_idx] -= PI / 2.0;
                let probs_minus = self.generator.get_probabilities_with_params(&params_minus);

                // Compute generator loss for each shift:
                // Loss = E_{x~G}[log(1 - D(x))]
                // We want to minimize this (fool discriminator), so gradient ascent on D(G(x))
                let loss_plus = self.evaluate_generator_loss(&probs_plus);
                let loss_minus = self.evaluate_generator_loss(&probs_minus);

                // Parameter-shift gradient
                let gradient = (loss_plus - loss_minus) / 2.0;

                // Update parameter (gradient descent on generator loss)
                self.generator.params[p_idx] -= self.gen_learning_rate * gradient;
            }

            // Compute current generator loss for reporting
            let current_probs = self.generator.get_probabilities();
            g_loss = self.evaluate_generator_loss(&current_probs);

            if epoch % 50 == 0 || epoch == epochs - 1 {
                println!(
                    "Epoch {}/{}: D_loss = {:.4}, G_loss = {:.4}",
                    epoch + 1,
                    epochs,
                    d_loss,
                    g_loss
                );
            }

            losses.push((d_loss, g_loss));
        }

        losses
    }

    /// Evaluate the generator loss given a probability distribution over bins.
    /// Loss = sum_x p(x) * log(1 - D(x))  (we want to minimize this)
    fn evaluate_generator_loss(&self, probs: &[f64]) -> f64 {
        let mut loss = 0.0;
        for (bin, &p) in probs.iter().enumerate() {
            if p < 1e-10 {
                continue;
            }
            let ret = self.bin_to_return(bin);
            let x = Array1::from_vec(vec![ret]);
            let d_output = self.discriminator.forward(&x);
            // Generator wants to maximize log(D(G(x)))
            // Equivalent to minimizing -log(D(G(x)))
            loss += p * (-(d_output + 1e-8).ln());
        }
        loss
    }

    /// Generate synthetic returns from the trained generator.
    pub fn generate_returns(&mut self, num_samples: usize) -> Vec<f64> {
        self.generator.sample_returns(num_samples, self.r_min, self.r_max)
    }
}

// =============================================================================
// Statistical comparison utilities
// =============================================================================

/// Compute the mean of a slice.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Compute the variance of a slice.
pub fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}

/// Compute the standard deviation.
pub fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Compute the skewness of a slice.
pub fn skewness(data: &[f64]) -> f64 {
    if data.len() < 3 {
        return 0.0;
    }
    let m = mean(data);
    let s = std_dev(data);
    if s < 1e-10 {
        return 0.0;
    }
    let n = data.len() as f64;
    let sum_cubed: f64 = data.iter().map(|&x| ((x - m) / s).powi(3)).sum();
    sum_cubed / n
}

/// Compute the kurtosis (excess) of a slice.
pub fn kurtosis(data: &[f64]) -> f64 {
    if data.len() < 4 {
        return 0.0;
    }
    let m = mean(data);
    let s = std_dev(data);
    if s < 1e-10 {
        return 0.0;
    }
    let n = data.len() as f64;
    let sum_fourth: f64 = data.iter().map(|&x| ((x - m) / s).powi(4)).sum();
    sum_fourth / n - 3.0
}

/// Print a statistical comparison between real and generated data.
pub fn compare_distributions(real: &[f64], generated: &[f64]) {
    println!("\n=== Statistical Comparison ===");
    println!("{:<15} {:>12} {:>12}", "Metric", "Real", "Generated");
    println!("{:-<39}", "");
    println!(
        "{:<15} {:>12.6} {:>12.6}",
        "Mean",
        mean(real),
        mean(generated)
    );
    println!(
        "{:<15} {:>12.6} {:>12.6}",
        "Variance",
        variance(real),
        variance(generated)
    );
    println!(
        "{:<15} {:>12.6} {:>12.6}",
        "Std Dev",
        std_dev(real),
        std_dev(generated)
    );
    println!(
        "{:<15} {:>12.6} {:>12.6}",
        "Skewness",
        skewness(real),
        skewness(generated)
    );
    println!(
        "{:<15} {:>12.6} {:>12.6}",
        "Kurtosis",
        kurtosis(real),
        kurtosis(generated)
    );
    println!("{:<15} {:>12} {:>12}", "Samples", real.len(), generated.len());
}

// =============================================================================
// Bybit API integration
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub list: Vec<Vec<String>>,
}

/// Fetch historical kline data from Bybit v5 API.
/// Returns close prices for the given symbol and interval.
///
/// Kline format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<f64>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let response: BybitResponse = client
        .get(&url)
        .header("User-Agent", "quantum-gan-finance/0.1.0")
        .send()?
        .json()?;

    if response.ret_code != 0 {
        anyhow::bail!(
            "Bybit API error: {} (code {})",
            response.ret_msg,
            response.ret_code
        );
    }

    let mut closes: Vec<f64> = Vec::new();
    for kline in &response.result.list {
        if kline.len() >= 5 {
            if let Ok(close) = kline[4].parse::<f64>() {
                if close > 0.0 {
                    closes.push(close);
                }
            }
        }
    }

    // Bybit returns data newest first, reverse for chronological order
    closes.reverse();

    Ok(closes)
}

/// Convert close prices to log returns.
pub fn compute_log_returns(closes: &[f64]) -> Vec<f64> {
    closes
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .filter(|r| r.is_finite() && r.abs() < 0.5)
        .collect()
}

/// Generate synthetic financial data using the quantum GAN pipeline:
/// 1. Fetch real data from Bybit.
/// 2. Train the QGAN on the real return distribution.
/// 3. Generate synthetic returns.
/// 4. Compare statistics.
pub fn run_qgan_pipeline(
    symbol: &str,
    num_qubits: usize,
    num_layers: usize,
    epochs: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    println!("Fetching {} data from Bybit...", symbol);
    let closes = fetch_bybit_klines(symbol, "60", 200)?;
    println!("Fetched {} close prices.", closes.len());

    let returns = compute_log_returns(&closes);
    println!("Computed {} log returns.", returns.len());

    if returns.is_empty() {
        anyhow::bail!("No valid returns computed from price data.");
    }

    // Determine return range
    let r_min = returns.iter().cloned().fold(f64::INFINITY, f64::min) * 1.5;
    let r_max = returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 1.5;
    println!("Return range: [{:.6}, {:.6}]", r_min, r_max);

    // Create and train QGAN
    println!(
        "\nTraining Quantum GAN ({} qubits, {} layers, {} epochs)...",
        num_qubits, num_layers, epochs
    );
    let mut qgan = QuantumGAN::new(num_qubits, num_layers, r_min, r_max);
    qgan.train(&returns, epochs, 32);

    // Generate synthetic returns
    let num_synthetic = returns.len();
    println!("\nGenerating {} synthetic returns...", num_synthetic);
    let synthetic = qgan.generate_returns(num_synthetic);

    // Compare distributions
    compare_distributions(&returns, &synthetic);

    Ok((returns, synthetic))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_circuit_probabilities_sum_to_one() {
        let mut circuit = QuantumCircuit::new(3, 2);
        let probs = circuit.get_probabilities();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Probabilities sum to {}", sum);
    }

    #[test]
    fn test_quantum_circuit_sampling() {
        let mut circuit = QuantumCircuit::new(3, 2);
        let samples = circuit.sample(100);
        assert_eq!(samples.len(), 100);
        for &s in &samples {
            assert!(s < 8, "Sample {} out of range for 3 qubits", s);
        }
    }

    #[test]
    fn test_discriminator_output_range() {
        let disc = Discriminator::new(1, 8, 0.01);
        let x = Array1::from_vec(vec![0.5]);
        let out = disc.forward(&x);
        assert!(out >= 0.0 && out <= 1.0, "Output {} not in [0,1]", out);
    }

    #[test]
    fn test_statistical_functions() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&data) - 3.0).abs() < 1e-10);
        assert!((variance(&data) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_log_returns() {
        let closes = vec![100.0, 101.0, 99.5, 102.0];
        let returns = compute_log_returns(&closes);
        assert_eq!(returns.len(), 3);
        assert!((returns[0] - (101.0_f64 / 100.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_discretize_returns() {
        let qgan = QuantumGAN::new(3, 2, -0.1, 0.1);
        let returns = vec![0.0, 0.01, -0.01, 0.05, -0.05];
        let hist = qgan.discretize_returns(&returns);
        let sum: f64 = hist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Histogram sums to {}", sum);
    }
}
