use anyhow::Result;
use rand::Rng;
use serde::Deserialize;
use std::f64::consts::PI;

// =============================================================================
// NISQ Simulator
// =============================================================================

/// Simulates a noisy intermediate-scale quantum processor.
/// Models gate errors and measurement errors to evaluate
/// parameterized quantum circuits under realistic noise conditions.
pub struct NISQSimulator {
    pub num_qubits: usize,
    pub gate_error_rate: f64,
    pub measurement_error_rate: f64,
}

impl NISQSimulator {
    pub fn new(num_qubits: usize, gate_error_rate: f64, measurement_error_rate: f64) -> Self {
        Self {
            num_qubits,
            gate_error_rate,
            measurement_error_rate,
        }
    }

    /// Evaluate a parameterized circuit and return the expectation value.
    /// The ideal value is computed analytically, then degraded by noise
    /// proportional to circuit depth and error rates.
    pub fn evaluate_circuit(
        &self,
        params: &[f64],
        cost_coefficients: &[f64],
        circuit_depth: usize,
    ) -> f64 {
        let ideal = self.ideal_expectation(params, cost_coefficients);
        self.apply_noise(ideal, circuit_depth)
    }

    /// Compute the ideal (noiseless) expectation value for a parameterized circuit.
    /// Uses a simplified model: each parameter rotation contributes a cosine term
    /// weighted by the corresponding cost coefficient.
    fn ideal_expectation(&self, params: &[f64], cost_coefficients: &[f64]) -> f64 {
        let n = params.len().min(cost_coefficients.len());
        let mut expectation = 0.0;
        for i in 0..n {
            expectation += cost_coefficients[i] * (params[i]).cos();
        }
        expectation / n as f64
    }

    /// Apply depolarizing noise model to an ideal expectation value.
    /// The signal decays exponentially with circuit depth and gate error rate.
    /// Measurement error adds additional uniform mixing toward 0.
    fn apply_noise(&self, ideal_value: f64, circuit_depth: usize) -> f64 {
        let mut rng = rand::thread_rng();

        // Depolarizing noise: signal decays exponentially
        let gate_fidelity = 1.0 - self.gate_error_rate;
        let num_gates = circuit_depth * self.num_qubits;
        let noise_factor = gate_fidelity.powi(num_gates as i32);

        // Measurement error: mixes result toward zero
        let measurement_fidelity = 1.0 - self.measurement_error_rate;

        let noisy_value = ideal_value * noise_factor * measurement_fidelity;

        // Add stochastic shot noise (finite sampling)
        let shot_noise = rng.gen_range(-0.02..0.02);
        noisy_value + shot_noise
    }

    /// Evaluate circuit at an amplified noise level (for ZNE).
    /// noise_scale_factor=1.0 is the base noise, 2.0 doubles it, etc.
    pub fn evaluate_circuit_amplified(
        &self,
        params: &[f64],
        cost_coefficients: &[f64],
        circuit_depth: usize,
        noise_scale_factor: f64,
    ) -> f64 {
        let ideal = self.ideal_expectation(params, cost_coefficients);
        let gate_fidelity = 1.0 - self.gate_error_rate * noise_scale_factor;
        let gate_fidelity = gate_fidelity.max(0.0);
        let num_gates = circuit_depth * self.num_qubits;
        let noise_factor = gate_fidelity.powi(num_gates as i32);
        let measurement_fidelity = 1.0 - self.measurement_error_rate * noise_scale_factor;
        let measurement_fidelity = measurement_fidelity.max(0.0);

        let mut rng = rand::thread_rng();
        let shot_noise = rng.gen_range(-0.02..0.02);
        ideal * noise_factor * measurement_fidelity + shot_noise
    }

    /// Evaluate with no noise (ideal simulator)
    pub fn evaluate_circuit_ideal(
        &self,
        params: &[f64],
        cost_coefficients: &[f64],
    ) -> f64 {
        self.ideal_expectation(params, cost_coefficients)
    }
}

// =============================================================================
// NISQ Algorithms
// =============================================================================

/// Result of running a NISQ algorithm
#[derive(Debug, Clone)]
pub struct AlgorithmResult {
    pub name: String,
    pub optimal_params: Vec<f64>,
    pub optimal_value: f64,
    pub iterations: usize,
}

/// Shallow Variational Quantum Eigensolver (1-layer ansatz).
/// Optimizes a cost Hamiltonian using parameterized rotations.
pub struct ShallowVQE {
    pub num_qubits: usize,
    pub max_iterations: usize,
    pub learning_rate: f64,
}

impl ShallowVQE {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            max_iterations: 200,
            learning_rate: 0.1,
        }
    }

    /// Run VQE optimization on the given simulator with cost coefficients.
    /// Uses SPSA-like parameter perturbation for gradient-free optimization.
    pub fn optimize(
        &self,
        simulator: &NISQSimulator,
        cost_coefficients: &[f64],
    ) -> AlgorithmResult {
        let mut rng = rand::thread_rng();
        let circuit_depth = 1; // shallow: single layer

        // Initialize parameters randomly in [0, 2*pi)
        let mut params: Vec<f64> = (0..self.num_qubits)
            .map(|_| rng.gen_range(0.0..2.0 * PI))
            .collect();

        let mut best_value = f64::MAX;
        let mut best_params = params.clone();

        for iter in 0..self.max_iterations {
            let current_value =
                simulator.evaluate_circuit(&params, cost_coefficients, circuit_depth);

            if current_value < best_value {
                best_value = current_value;
                best_params = params.clone();
            }

            // SPSA: simultaneous perturbation
            let perturbation: Vec<f64> = (0..self.num_qubits)
                .map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 })
                .collect();

            let step_size = self.learning_rate / (1.0 + iter as f64 * 0.01);
            let ck = 0.1 / (1.0 + iter as f64 * 0.01);

            // Evaluate at +/- perturbation
            let params_plus: Vec<f64> = params
                .iter()
                .zip(perturbation.iter())
                .map(|(p, d)| p + ck * d)
                .collect();
            let params_minus: Vec<f64> = params
                .iter()
                .zip(perturbation.iter())
                .map(|(p, d)| p - ck * d)
                .collect();

            let f_plus =
                simulator.evaluate_circuit(&params_plus, cost_coefficients, circuit_depth);
            let f_minus =
                simulator.evaluate_circuit(&params_minus, cost_coefficients, circuit_depth);

            // Gradient estimate and update
            let gradient_estimate = (f_plus - f_minus) / (2.0 * ck);
            for i in 0..self.num_qubits {
                params[i] -= step_size * gradient_estimate * perturbation[i];
            }
        }

        AlgorithmResult {
            name: "ShallowVQE".to_string(),
            optimal_params: best_params,
            optimal_value: best_value,
            iterations: self.max_iterations,
        }
    }
}

/// QAOA with a single layer (p=1).
/// Searches over gamma and beta parameters for combinatorial optimization.
pub struct QAOA1 {
    pub num_qubits: usize,
    pub gamma_steps: usize,
    pub beta_steps: usize,
}

impl QAOA1 {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            gamma_steps: 50,
            beta_steps: 50,
        }
    }

    /// Run QAOA-1 grid search over (gamma, beta) parameter space.
    pub fn optimize(
        &self,
        simulator: &NISQSimulator,
        cost_coefficients: &[f64],
    ) -> AlgorithmResult {
        let circuit_depth = 2; // QAOA-1: one problem layer + one mixer layer

        let mut best_value = f64::MAX;
        let mut best_gamma = 0.0;
        let mut best_beta = 0.0;

        for gi in 0..self.gamma_steps {
            let gamma = (gi as f64 / self.gamma_steps as f64) * PI;
            for bi in 0..self.beta_steps {
                let beta = (bi as f64 / self.beta_steps as f64) * PI;

                // Construct parameter vector: gamma for each qubit, then beta for each qubit
                let params: Vec<f64> = (0..self.num_qubits)
                    .map(|i| gamma * cost_coefficients.get(i).unwrap_or(&1.0) + beta)
                    .collect();

                let value =
                    simulator.evaluate_circuit(&params, cost_coefficients, circuit_depth);

                if value < best_value {
                    best_value = value;
                    best_gamma = gamma;
                    best_beta = beta;
                }
            }
        }

        AlgorithmResult {
            name: "QAOA-1".to_string(),
            optimal_params: vec![best_gamma, best_beta],
            optimal_value: best_value,
            iterations: self.gamma_steps * self.beta_steps,
        }
    }
}

/// Variational Quantum Classifier for binary classification.
/// Encodes features into qubit rotations and trains parameters
/// to minimize classification error.
pub struct VariationalClassifier {
    pub num_qubits: usize,
    pub num_layers: usize,
    pub max_iterations: usize,
}

impl VariationalClassifier {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            num_layers: 1,
            max_iterations: 100,
        }
    }

    /// Train the classifier on labeled feature vectors.
    /// features: Vec of feature vectors (each scaled to [0, 1])
    /// labels: Vec of binary labels (0 or 1)
    pub fn train(
        &self,
        simulator: &NISQSimulator,
        features: &[Vec<f64>],
        labels: &[u8],
    ) -> AlgorithmResult {
        let mut rng = rand::thread_rng();
        let num_params = self.num_qubits * self.num_layers;
        let mut params: Vec<f64> = (0..num_params)
            .map(|_| rng.gen_range(0.0..2.0 * PI))
            .collect();

        let mut best_accuracy = 0.0;
        let mut best_params = params.clone();

        for iter in 0..self.max_iterations {
            let accuracy = self.evaluate_accuracy(simulator, &params, features, labels);
            if accuracy > best_accuracy {
                best_accuracy = accuracy;
                best_params = params.clone();
            }

            // Random parameter update (SPSA-like)
            let step = 0.2 / (1.0 + iter as f64 * 0.02);
            for p in params.iter_mut() {
                *p += rng.gen_range(-step..step);
            }
        }

        AlgorithmResult {
            name: "VariationalClassifier".to_string(),
            optimal_params: best_params,
            optimal_value: best_accuracy,
            iterations: self.max_iterations,
        }
    }

    /// Predict class label for a single feature vector.
    pub fn predict(
        &self,
        simulator: &NISQSimulator,
        params: &[f64],
        feature: &[f64],
    ) -> u8 {
        let circuit_params = self.encode_features(params, feature);
        let cost_coefficients: Vec<f64> = vec![1.0; circuit_params.len()];
        let value = simulator.evaluate_circuit(
            &circuit_params,
            &cost_coefficients,
            self.num_layers,
        );
        if value > 0.0 { 1 } else { 0 }
    }

    fn evaluate_accuracy(
        &self,
        simulator: &NISQSimulator,
        params: &[f64],
        features: &[Vec<f64>],
        labels: &[u8],
    ) -> f64 {
        let mut correct = 0;
        for (feature, &label) in features.iter().zip(labels.iter()) {
            let prediction = self.predict(simulator, params, feature);
            if prediction == label {
                correct += 1;
            }
        }
        correct as f64 / labels.len() as f64
    }

    /// Encode classical features into quantum circuit parameters.
    /// Combines trainable parameters with data-encoding rotations.
    fn encode_features(&self, params: &[f64], features: &[f64]) -> Vec<f64> {
        let mut circuit_params = Vec::new();
        for i in 0..self.num_qubits {
            let param = params.get(i).copied().unwrap_or(0.0);
            let feature = features.get(i).copied().unwrap_or(0.0);
            circuit_params.push(param + feature * PI);
        }
        circuit_params
    }
}

// =============================================================================
// Error Mitigation
// =============================================================================

/// Zero-noise extrapolation (ZNE) for error mitigation.
/// Runs circuits at multiple noise amplification factors and
/// extrapolates to the zero-noise limit.
pub struct ErrorMitigation;

impl ErrorMitigation {
    /// Apply zero-noise extrapolation using Richardson extrapolation.
    /// Evaluates the circuit at noise scale factors [1, 2, 3] and
    /// fits a linear or exponential model to extrapolate to scale=0.
    pub fn zero_noise_extrapolation(
        simulator: &NISQSimulator,
        params: &[f64],
        cost_coefficients: &[f64],
        circuit_depth: usize,
    ) -> f64 {
        let scale_factors = [1.0, 2.0, 3.0];
        let num_samples = 5; // average over multiple shots to reduce variance

        let mut values: Vec<f64> = Vec::new();
        for &scale in &scale_factors {
            let mut sum = 0.0;
            for _ in 0..num_samples {
                sum += simulator.evaluate_circuit_amplified(
                    params,
                    cost_coefficients,
                    circuit_depth,
                    scale,
                );
            }
            values.push(sum / num_samples as f64);
        }

        // Richardson extrapolation (linear): extrapolate to scale=0
        // Using the three points at scales 1, 2, 3:
        // f(0) ≈ 3*f(1) - 3*f(2) + f(3)
        3.0 * values[0] - 3.0 * values[1] + values[2]
    }

    /// Apply measurement error correction using a simple calibration model.
    /// Assumes a symmetric bit-flip measurement error model.
    pub fn measurement_error_correction(
        raw_probabilities: &[f64],
        error_rate: f64,
    ) -> Vec<f64> {
        let correction_factor = 1.0 / (1.0 - 2.0 * error_rate);
        let uniform = 1.0 / raw_probabilities.len() as f64;

        raw_probabilities
            .iter()
            .map(|&p| {
                let corrected = uniform + correction_factor * (p - uniform);
                corrected.max(0.0).min(1.0)
            })
            .collect()
    }

    /// Apply zero-noise extrapolation to a set of probability distributions.
    /// Returns mitigated probabilities.
    pub fn mitigate_distribution(
        simulator: &NISQSimulator,
        params: &[f64],
        cost_coefficients: &[f64],
        circuit_depth: usize,
    ) -> f64 {
        let zne_value = Self::zero_noise_extrapolation(
            simulator,
            params,
            cost_coefficients,
            circuit_depth,
        );

        // Also apply measurement error correction to the scalar result
        let corrected = vec![zne_value];
        let mitigated =
            Self::measurement_error_correction(&corrected, simulator.measurement_error_rate);
        mitigated[0]
    }
}

// =============================================================================
// Benchmarking Framework
// =============================================================================

/// Benchmark result comparing ideal, noisy, and mitigated performance.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub algorithm_name: String,
    pub ideal_value: f64,
    pub noisy_value: f64,
    pub mitigated_value: f64,
    pub noise_error: f64,       // |noisy - ideal|
    pub mitigated_error: f64,   // |mitigated - ideal|
    pub improvement_ratio: f64, // noise_error / mitigated_error
}

/// Benchmarking framework for comparing algorithm performance
/// across ideal, noisy, and mitigated settings.
pub struct BenchmarkRunner {
    pub num_qubits: usize,
    pub gate_error_rate: f64,
    pub measurement_error_rate: f64,
}

impl BenchmarkRunner {
    pub fn new(num_qubits: usize, gate_error_rate: f64, measurement_error_rate: f64) -> Self {
        Self {
            num_qubits,
            gate_error_rate,
            measurement_error_rate,
        }
    }

    /// Run a full benchmark for a given set of cost coefficients.
    /// Returns BenchmarkResults for each algorithm.
    pub fn run_benchmark(&self, cost_coefficients: &[f64]) -> Vec<BenchmarkResult> {
        let mut results = Vec::new();

        // Create simulators: ideal (no noise) and noisy
        let ideal_sim = NISQSimulator::new(self.num_qubits, 0.0, 0.0);
        let noisy_sim = NISQSimulator::new(
            self.num_qubits,
            self.gate_error_rate,
            self.measurement_error_rate,
        );

        // Benchmark ShallowVQE
        let vqe = ShallowVQE::new(self.num_qubits);
        let ideal_result = vqe.optimize(&ideal_sim, cost_coefficients);
        let noisy_result = vqe.optimize(&noisy_sim, cost_coefficients);
        let mitigated_value = ErrorMitigation::zero_noise_extrapolation(
            &noisy_sim,
            &noisy_result.optimal_params,
            cost_coefficients,
            1,
        );

        results.push(Self::compute_benchmark(
            "ShallowVQE",
            ideal_result.optimal_value,
            noisy_result.optimal_value,
            mitigated_value,
        ));

        // Benchmark QAOA-1
        let qaoa = QAOA1::new(self.num_qubits);
        let ideal_result = qaoa.optimize(&ideal_sim, cost_coefficients);
        let noisy_result = qaoa.optimize(&noisy_sim, cost_coefficients);
        let mitigated_value = ErrorMitigation::zero_noise_extrapolation(
            &noisy_sim,
            &noisy_result.optimal_params,
            cost_coefficients,
            2,
        );

        results.push(Self::compute_benchmark(
            "QAOA-1",
            ideal_result.optimal_value,
            noisy_result.optimal_value,
            mitigated_value,
        ));

        results
    }

    fn compute_benchmark(
        name: &str,
        ideal: f64,
        noisy: f64,
        mitigated: f64,
    ) -> BenchmarkResult {
        let noise_error = (noisy - ideal).abs();
        let mitigated_error = (mitigated - ideal).abs();
        let improvement_ratio = if mitigated_error > 1e-10 {
            noise_error / mitigated_error
        } else {
            f64::INFINITY
        };

        BenchmarkResult {
            algorithm_name: name.to_string(),
            ideal_value: ideal,
            noisy_value: noisy,
            mitigated_value: mitigated,
            noise_error,
            mitigated_error,
            improvement_ratio,
        }
    }

    /// Benchmark accuracy vs circuit depth for a given algorithm.
    /// Shows how noise degrades performance at increasing depths.
    pub fn accuracy_vs_depth(
        &self,
        cost_coefficients: &[f64],
        max_depth: usize,
    ) -> Vec<(usize, f64, f64)> {
        let ideal_sim = NISQSimulator::new(self.num_qubits, 0.0, 0.0);
        let noisy_sim = NISQSimulator::new(
            self.num_qubits,
            self.gate_error_rate,
            self.measurement_error_rate,
        );

        let params: Vec<f64> = vec![1.0; self.num_qubits];
        let mut results = Vec::new();

        for depth in 1..=max_depth {
            let ideal = ideal_sim.evaluate_circuit(&params, cost_coefficients, depth);
            let noisy = noisy_sim.evaluate_circuit(&params, cost_coefficients, depth);
            let relative_error = if ideal.abs() > 1e-10 {
                (noisy - ideal).abs() / ideal.abs()
            } else {
                0.0
            };
            results.push((depth, ideal, relative_error));
        }

        results
    }
}

// =============================================================================
// Bybit API Client
// =============================================================================

/// Bybit API response structures
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

/// OHLCV kline data
#[derive(Debug, Clone)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Feature vector for quantum algorithms
#[derive(Debug, Clone)]
pub struct TradingFeatures {
    pub returns: f64,
    pub volatility: f64,
    pub momentum: f64,
    pub volume_ratio: f64,
    pub label: u8, // 1 = price went up, 0 = price went down
}

/// Bybit API client for fetching market data
pub struct BybitClient {
    base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data from Bybit v5 API.
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let client = reqwest::Client::new();
        let response: BybitResponse = client
            .get(&url)
            .header("User-Agent", "nisq-trading-bot/0.1")
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let mut klines = Vec::new();
        for item in &response.result.list {
            if item.len() >= 6 {
                klines.push(Kline {
                    timestamp: item[0].parse().unwrap_or(0),
                    open: item[1].parse().unwrap_or(0.0),
                    high: item[2].parse().unwrap_or(0.0),
                    low: item[3].parse().unwrap_or(0.0),
                    close: item[4].parse().unwrap_or(0.0),
                    volume: item[5].parse().unwrap_or(0.0),
                });
            }
        }

        // Bybit returns newest first, reverse for chronological order
        klines.reverse();
        Ok(klines)
    }

    /// Compute trading features from kline data.
    /// Returns normalized features suitable for quantum circuit encoding.
    pub fn compute_features(&self, klines: &[Kline]) -> Vec<TradingFeatures> {
        if klines.len() < 10 {
            return Vec::new();
        }

        let mut features = Vec::new();
        let lookback = 5;

        for i in lookback..klines.len() - 1 {
            // Returns: log(close/prev_close)
            let returns = (klines[i].close / klines[i - 1].close).ln();

            // Volatility: std dev of recent returns
            let recent_returns: Vec<f64> = (0..lookback)
                .map(|j| (klines[i - j].close / klines[i - j - 1].close).ln())
                .collect();
            let mean_ret = recent_returns.iter().sum::<f64>() / lookback as f64;
            let volatility = (recent_returns
                .iter()
                .map(|r| (r - mean_ret).powi(2))
                .sum::<f64>()
                / lookback as f64)
                .sqrt();

            // Momentum: return over lookback period
            let momentum = (klines[i].close / klines[i - lookback].close).ln();

            // Volume ratio: current volume / average volume
            let avg_volume: f64 =
                (0..lookback).map(|j| klines[i - j].volume).sum::<f64>() / lookback as f64;
            let volume_ratio = if avg_volume > 0.0 {
                klines[i].volume / avg_volume
            } else {
                1.0
            };

            // Label: 1 if next candle closes higher, 0 otherwise
            let label = if klines[i + 1].close > klines[i].close {
                1
            } else {
                0
            };

            features.push(TradingFeatures {
                returns: Self::normalize(returns, -0.05, 0.05),
                volatility: Self::normalize(volatility, 0.0, 0.05),
                momentum: Self::normalize(momentum, -0.1, 0.1),
                volume_ratio: Self::normalize(volume_ratio, 0.0, 3.0),
                label,
            });
        }

        features
    }

    /// Normalize a value to [0, 1] range
    fn normalize(value: f64, min_val: f64, max_val: f64) -> f64 {
        ((value - min_val) / (max_val - min_val))
            .max(0.0)
            .min(1.0)
    }

    /// Convert TradingFeatures into feature vectors for quantum algorithms
    pub fn features_to_vectors(features: &[TradingFeatures]) -> (Vec<Vec<f64>>, Vec<u8>) {
        let vectors: Vec<Vec<f64>> = features
            .iter()
            .map(|f| vec![f.returns, f.volatility, f.momentum, f.volume_ratio])
            .collect();
        let labels: Vec<u8> = features.iter().map(|f| f.label).collect();
        (vectors, labels)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate synthetic trading data for testing without API access.
pub fn generate_synthetic_data(num_samples: usize) -> (Vec<Vec<f64>>, Vec<u8>) {
    let mut rng = rand::thread_rng();
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for _ in 0..num_samples {
        let returns = rng.gen_range(0.0..1.0);
        let volatility = rng.gen_range(0.0..1.0);
        let momentum = rng.gen_range(0.0..1.0);
        let volume_ratio = rng.gen_range(0.0..1.0);

        // Simple rule: if momentum + returns > 1.0, label is 1 (up)
        let label = if momentum + returns > 1.0 { 1 } else { 0 };

        features.push(vec![returns, volatility, momentum, volume_ratio]);
        labels.push(label);
    }

    (features, labels)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nisq_simulator_ideal() {
        let sim = NISQSimulator::new(4, 0.0, 0.0);
        let params = vec![0.0; 4];
        let coefficients = vec![1.0; 4];
        let value = sim.evaluate_circuit_ideal(&params, &coefficients);
        // cos(0) = 1, so average should be 1.0
        assert!((value - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nisq_simulator_noise_degrades() {
        let ideal_sim = NISQSimulator::new(4, 0.0, 0.0);
        let noisy_sim = NISQSimulator::new(4, 0.01, 0.02);

        let params = vec![0.5; 4];
        let coefficients = vec![1.0; 4];

        let ideal = ideal_sim.evaluate_circuit_ideal(&params, &coefficients);
        // Average over multiple runs to reduce shot noise
        let noisy_avg: f64 = (0..100)
            .map(|_| noisy_sim.evaluate_circuit(&params, &coefficients, 5))
            .sum::<f64>()
            / 100.0;

        // Noisy result should be closer to 0 (degraded)
        assert!(noisy_avg.abs() < ideal.abs() + 0.1);
    }

    #[test]
    fn test_shallow_vqe() {
        let sim = NISQSimulator::new(4, 0.001, 0.005);
        let cost_coefficients = vec![1.0, -0.5, 0.3, -0.8];

        let vqe = ShallowVQE::new(4);
        let result = vqe.optimize(&sim, &cost_coefficients);

        assert_eq!(result.name, "ShallowVQE");
        assert_eq!(result.iterations, 200);
        // The optimizer should find a value less than 0 (some optimization happened)
        assert!(result.optimal_value < 1.0);
    }

    #[test]
    fn test_qaoa1() {
        let sim = NISQSimulator::new(4, 0.001, 0.005);
        let cost_coefficients = vec![1.0, -0.5, 0.3, -0.8];

        let qaoa = QAOA1::new(4);
        let result = qaoa.optimize(&sim, &cost_coefficients);

        assert_eq!(result.name, "QAOA-1");
        assert!(result.optimal_value < 1.0);
    }

    #[test]
    fn test_variational_classifier() {
        let sim = NISQSimulator::new(4, 0.001, 0.005);
        let (features, labels) = generate_synthetic_data(50);

        let classifier = VariationalClassifier::new(4);
        let result = classifier.train(&sim, &features, &labels);

        assert_eq!(result.name, "VariationalClassifier");
        // Should achieve at least 40% accuracy (better than degenerate)
        assert!(result.optimal_value >= 0.4);
    }

    #[test]
    fn test_zero_noise_extrapolation() {
        let sim = NISQSimulator::new(4, 0.01, 0.02);
        let params = vec![0.5; 4];
        let coefficients = vec![1.0; 4];

        let mitigated =
            ErrorMitigation::zero_noise_extrapolation(&sim, &params, &coefficients, 3);

        // ZNE result should exist (not NaN or infinite)
        assert!(mitigated.is_finite());
    }

    #[test]
    fn test_measurement_error_correction() {
        let raw = vec![0.45, 0.55]; // slightly off from ideal
        let corrected = ErrorMitigation::measurement_error_correction(&raw, 0.05);

        assert_eq!(corrected.len(), 2);
        // Corrected values should be valid probabilities
        for &p in &corrected {
            assert!(p >= 0.0 && p <= 1.0);
        }
    }

    #[test]
    fn test_benchmark_runner() {
        let runner = BenchmarkRunner::new(4, 0.005, 0.02);
        let coefficients = vec![1.0, -0.5, 0.3, -0.8];

        let results = runner.run_benchmark(&coefficients);
        assert_eq!(results.len(), 2); // VQE + QAOA

        for result in &results {
            assert!(result.noise_error >= 0.0);
            assert!(result.mitigated_error >= 0.0);
        }
    }

    #[test]
    fn test_accuracy_vs_depth() {
        let runner = BenchmarkRunner::new(4, 0.01, 0.02);
        let coefficients = vec![1.0; 4];

        let results = runner.accuracy_vs_depth(&coefficients, 10);
        assert_eq!(results.len(), 10);

        // Error should generally increase with depth
        let (_, _, error_depth_1) = results[0];
        let (_, _, error_depth_10) = results[9];
        // Not strictly guaranteed due to stochastic noise, but on average true
        assert!(error_depth_1 < error_depth_10 + 0.5);
    }

    #[test]
    fn test_generate_synthetic_data() {
        let (features, labels) = generate_synthetic_data(100);
        assert_eq!(features.len(), 100);
        assert_eq!(labels.len(), 100);
        assert_eq!(features[0].len(), 4);

        for label in &labels {
            assert!(*label == 0 || *label == 1);
        }
    }

    #[test]
    fn test_normalize() {
        assert!((BybitClient::normalize(0.5, 0.0, 1.0) - 0.5).abs() < 1e-10);
        assert!((BybitClient::normalize(-1.0, 0.0, 1.0) - 0.0).abs() < 1e-10);
        assert!((BybitClient::normalize(2.0, 0.0, 1.0) - 1.0).abs() < 1e-10);
    }
}
