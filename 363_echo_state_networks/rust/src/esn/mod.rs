//! Echo State Network Core Implementation
//!
//! This module provides the core ESN implementation including:
//! - Basic ESN with configurable parameters
//! - Deep ESN (stacked reservoirs)
//! - Ensemble ESN (multiple models)
//! - Online learning support

mod reservoir;
mod training;
mod prediction;

pub use reservoir::{Reservoir, ReservoirConfig};
pub use training::TrainingMethod;
pub use prediction::OnlineLearner;

use ndarray::{Array1, Array2, Axis, concatenate, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for Echo State Network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ESNConfig {
    /// Number of input features
    pub input_dim: usize,
    /// Number of neurons in the reservoir
    pub reservoir_size: usize,
    /// Number of output dimensions
    pub output_dim: usize,
    /// Spectral radius of reservoir matrix (should be < 1)
    pub spectral_radius: f64,
    /// Leaking rate for leaky integrator neurons (0 to 1)
    pub leaking_rate: f64,
    /// Scaling factor for input weights
    pub input_scaling: f64,
    /// Sparsity of reservoir connections (0 to 1)
    pub sparsity: f64,
    /// Regularization parameter for ridge regression
    pub regularization: f64,
    /// Number of initial timesteps to discard (washout)
    pub washout: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for ESNConfig {
    fn default() -> Self {
        Self {
            input_dim: 1,
            reservoir_size: 500,
            output_dim: 1,
            spectral_radius: 0.95,
            leaking_rate: 0.3,
            input_scaling: 0.1,
            sparsity: 0.1,
            regularization: 1e-6,
            washout: 100,
            seed: None,
        }
    }
}

impl ESNConfig {
    /// Create a new configuration with default values
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            ..Default::default()
        }
    }

    /// Set the reservoir size
    pub fn reservoir_size(mut self, size: usize) -> Self {
        self.reservoir_size = size;
        self
    }

    /// Set the spectral radius
    pub fn spectral_radius(mut self, radius: f64) -> Self {
        self.spectral_radius = radius;
        self
    }

    /// Set the leaking rate
    pub fn leaking_rate(mut self, rate: f64) -> Self {
        self.leaking_rate = rate;
        self
    }

    /// Set the input scaling
    pub fn input_scaling(mut self, scaling: f64) -> Self {
        self.input_scaling = scaling;
        self
    }

    /// Set the sparsity
    pub fn sparsity(mut self, sparsity: f64) -> Self {
        self.sparsity = sparsity;
        self
    }

    /// Set the regularization parameter
    pub fn regularization(mut self, reg: f64) -> Self {
        self.regularization = reg;
        self
    }

    /// Set the washout period
    pub fn washout(mut self, washout: usize) -> Self {
        self.washout = washout;
        self
    }

    /// Set the random seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Echo State Network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EchoStateNetwork {
    /// Configuration
    pub config: ESNConfig,
    /// Input weight matrix (reservoir_size x input_dim)
    w_in: Array2<f64>,
    /// Reservoir weight matrix (reservoir_size x reservoir_size)
    w_res: Array2<f64>,
    /// Output weight matrix (output_dim x (reservoir_size + input_dim))
    w_out: Array2<f64>,
    /// Current reservoir state
    state: Array1<f64>,
    /// Whether the network has been trained
    trained: bool,
}

impl EchoStateNetwork {
    /// Create a new Echo State Network
    pub fn new(config: ESNConfig) -> Self {
        let mut rng = match config.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed).into(),
            None => rand::thread_rng(),
        };

        // Initialize input weights
        let w_in = create_input_weights(
            config.reservoir_size,
            config.input_dim,
            config.input_scaling,
            &mut rng,
        );

        // Initialize reservoir weights
        let w_res = create_reservoir_weights(
            config.reservoir_size,
            config.sparsity,
            config.spectral_radius,
            &mut rng,
        );

        // Initialize output weights (will be trained)
        let w_out = Array2::zeros((config.output_dim, config.reservoir_size + config.input_dim));

        // Initialize state
        let state = Array1::zeros(config.reservoir_size);

        Self {
            config,
            w_in,
            w_res,
            w_out,
            state,
            trained: false,
        }
    }

    /// Reset the reservoir state to zeros
    pub fn reset_state(&mut self) {
        self.state.fill(0.0);
    }

    /// Get the current reservoir state
    pub fn get_state(&self) -> &Array1<f64> {
        &self.state
    }

    /// Update the reservoir state with a new input
    pub fn update(&mut self, input: &Array1<f64>) -> Array1<f64> {
        // Compute pre-activation: W_in * u + W_res * x
        let pre_activation = self.w_in.dot(input) + self.w_res.dot(&self.state);

        // Apply leaky integration with tanh activation
        self.state = &self.state * (1.0 - self.config.leaking_rate)
            + pre_activation.mapv(|x| x.tanh()) * self.config.leaking_rate;

        self.state.clone()
    }

    /// Get prediction from current state (without updating)
    pub fn predict(&self, input: &Array1<f64>) -> Array1<f64> {
        // Concatenate input and state
        let extended_state = concatenate![Axis(0), input.clone(), self.state.clone()];

        // Compute output
        self.w_out.dot(&extended_state)
    }

    /// Update state and get prediction in one step
    pub fn step(&mut self, input: &Array1<f64>) -> Array1<f64> {
        self.update(input);
        self.predict(input)
    }

    /// Train the output weights using ridge regression
    pub fn train(&mut self, inputs: &[Array1<f64>], targets: &[Array1<f64>]) {
        assert_eq!(inputs.len(), targets.len(), "Inputs and targets must have same length");
        assert!(inputs.len() > self.config.washout, "Not enough samples for washout");

        // Collect states (after washout period)
        let mut states = Vec::with_capacity(inputs.len() - self.config.washout);
        self.reset_state();

        for (i, input) in inputs.iter().enumerate() {
            self.update(input);

            if i >= self.config.washout {
                // Concatenate input and state
                let extended = concatenate![Axis(0), input.clone(), self.state.clone()];
                states.push(extended);
            }
        }

        // Build state matrix X (n_samples x state_dim)
        let n_samples = states.len();
        let state_dim = states[0].len();
        let mut x = Array2::zeros((n_samples, state_dim));
        for (i, state) in states.iter().enumerate() {
            x.row_mut(i).assign(state);
        }

        // Build target matrix Y (n_samples x output_dim)
        let targets_after_washout: Vec<_> = targets[self.config.washout..].to_vec();
        let mut y = Array2::zeros((n_samples, self.config.output_dim));
        for (i, target) in targets_after_washout.iter().enumerate() {
            y.row_mut(i).assign(target);
        }

        // Ridge regression: W_out = (X^T X + λI)^(-1) X^T Y
        let xt = x.t();
        let xtx = xt.dot(&x);
        let lambda_i = Array2::eye(state_dim) * self.config.regularization;
        let xtx_reg = &xtx + &lambda_i;

        // Use pseudo-inverse for numerical stability
        let xty = xt.dot(&y);

        // Solve the linear system
        if let Ok(w_out) = solve_linear_system(&xtx_reg, &xty) {
            self.w_out = w_out.t().to_owned();
        } else {
            // Fallback to simpler approach
            log::warn!("Linear system solve failed, using pseudo-inverse");
            let pinv = pseudo_inverse(&xtx_reg);
            self.w_out = pinv.dot(&xty).t().to_owned();
        }

        self.trained = true;
        self.reset_state();
    }

    /// Check if the network has been trained
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Save the model to a file
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let encoded = bincode::serialize(self)?;
        std::fs::write(path, encoded)?;
        Ok(())
    }

    /// Load a model from a file
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let data = std::fs::read(path)?;
        let model = bincode::deserialize(&data)?;
        Ok(model)
    }
}

/// Deep Echo State Network with stacked reservoirs
#[derive(Debug, Clone)]
pub struct DeepESN {
    layers: Vec<EchoStateNetwork>,
}

impl DeepESN {
    /// Create a new Deep ESN with specified layer sizes
    pub fn new(layer_configs: Vec<ESNConfig>) -> Self {
        let layers: Vec<_> = layer_configs.into_iter()
            .map(EchoStateNetwork::new)
            .collect();

        Self { layers }
    }

    /// Reset all layer states
    pub fn reset_state(&mut self) {
        for layer in &mut self.layers {
            layer.reset_state();
        }
    }

    /// Forward pass through all layers
    pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        let mut current = input.clone();

        for layer in &mut self.layers {
            layer.update(&current);
            current = layer.state.clone();
        }

        // Return prediction from last layer
        self.layers.last().unwrap().predict(input)
    }

    /// Get combined state from all layers
    pub fn get_combined_state(&self) -> Array1<f64> {
        let total_size: usize = self.layers.iter().map(|l| l.config.reservoir_size).sum();
        let mut combined = Array1::zeros(total_size);

        let mut offset = 0;
        for layer in &self.layers {
            let state = layer.get_state();
            combined.slice_mut(s![offset..offset + state.len()]).assign(state);
            offset += state.len();
        }

        combined
    }
}

/// Ensemble of Echo State Networks
#[derive(Debug, Clone)]
pub struct EnsembleESN {
    models: Vec<EchoStateNetwork>,
    weights: Vec<f64>,
}

impl EnsembleESN {
    /// Create a new ensemble with multiple ESNs
    pub fn new(config: ESNConfig, n_models: usize) -> Self {
        let models: Vec<_> = (0..n_models)
            .map(|i| {
                let mut cfg = config.clone();
                cfg.seed = Some(i as u64);
                EchoStateNetwork::new(cfg)
            })
            .collect();

        let weights = vec![1.0 / n_models as f64; n_models];

        Self { models, weights }
    }

    /// Reset all model states
    pub fn reset_state(&mut self) {
        for model in &mut self.models {
            model.reset_state();
        }
    }

    /// Train all models
    pub fn train(&mut self, inputs: &[Array1<f64>], targets: &[Array1<f64>]) {
        for model in &mut self.models {
            model.train(inputs, targets);
        }
    }

    /// Get ensemble prediction (weighted average)
    pub fn predict(&mut self, input: &Array1<f64>) -> Array1<f64> {
        let predictions: Vec<_> = self.models.iter_mut()
            .map(|m| m.step(input))
            .collect();

        // Weighted average
        let mut result = Array1::zeros(predictions[0].len());
        for (pred, weight) in predictions.iter().zip(&self.weights) {
            result = &result + &(pred * *weight);
        }

        result
    }

    /// Get prediction variance (uncertainty estimate)
    pub fn predict_with_uncertainty(&mut self, input: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let predictions: Vec<_> = self.models.iter_mut()
            .map(|m| m.step(input))
            .collect();

        // Mean
        let mut mean = Array1::zeros(predictions[0].len());
        for pred in &predictions {
            mean = &mean + pred;
        }
        mean = mean / self.models.len() as f64;

        // Variance
        let mut variance = Array1::zeros(predictions[0].len());
        for pred in &predictions {
            let diff = pred - &mean;
            variance = &variance + &(&diff * &diff);
        }
        variance = variance / self.models.len() as f64;

        (mean, variance)
    }
}

// Helper functions

fn create_input_weights<R: Rng>(
    reservoir_size: usize,
    input_dim: usize,
    scaling: f64,
    rng: &mut R,
) -> Array2<f64> {
    let mut w_in = Array2::random_using(
        (reservoir_size, input_dim),
        Uniform::new(-1.0, 1.0),
        rng,
    );
    w_in *= scaling;
    w_in
}

fn create_reservoir_weights<R: Rng>(
    size: usize,
    sparsity: f64,
    spectral_radius: f64,
    rng: &mut R,
) -> Array2<f64> {
    // Create sparse random matrix
    let mut w = Array2::zeros((size, size));
    let dist = Uniform::new(-1.0, 1.0);

    for i in 0..size {
        for j in 0..size {
            if rng.gen::<f64>() < sparsity {
                w[[i, j]] = rng.sample(dist);
            }
        }
    }

    // Scale to desired spectral radius
    let current_radius = estimate_spectral_radius(&w);
    if current_radius > 0.0 {
        w *= spectral_radius / current_radius;
    }

    w
}

fn estimate_spectral_radius(matrix: &Array2<f64>) -> f64 {
    // Power iteration to estimate largest eigenvalue
    let n = matrix.nrows();
    let mut v = Array1::random(n, Uniform::new(0.0, 1.0));
    v /= norm(&v);

    for _ in 0..100 {
        let v_new = matrix.dot(&v);
        let norm_new = norm(&v_new);
        if norm_new < 1e-10 {
            return 0.0;
        }
        v = v_new / norm_new;
    }

    let av = matrix.dot(&v);
    norm(&av) / norm(&v)
}

fn norm(v: &Array1<f64>) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn solve_linear_system(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, &'static str> {
    // Simple Gaussian elimination (for demonstration)
    // In production, use ndarray-linalg or nalgebra
    let pinv = pseudo_inverse(a);
    Ok(pinv.dot(b))
}

fn pseudo_inverse(matrix: &Array2<f64>) -> Array2<f64> {
    // Simple pseudo-inverse using (A^T A)^(-1) A^T
    let at = matrix.t();
    let ata = at.dot(matrix);

    // Add small regularization for stability
    let reg = Array2::eye(ata.nrows()) * 1e-10;
    let ata_reg = &ata + &reg;

    // Naive inverse (for demonstration)
    // In production, use proper linear algebra library
    matrix_inverse(&ata_reg).dot(&at.to_owned())
}

fn matrix_inverse(matrix: &Array2<f64>) -> Array2<f64> {
    // Gauss-Jordan elimination for matrix inverse
    let n = matrix.nrows();
    let mut augmented = Array2::zeros((n, 2 * n));

    // Set up augmented matrix [A | I]
    for i in 0..n {
        for j in 0..n {
            augmented[[i, j]] = matrix[[i, j]];
        }
        augmented[[i, n + i]] = 1.0;
    }

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if augmented[[k, i]].abs() > augmented[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows
        for j in 0..(2 * n) {
            let temp = augmented[[i, j]];
            augmented[[i, j]] = augmented[[max_row, j]];
            augmented[[max_row, j]] = temp;
        }

        // Scale pivot row
        let pivot = augmented[[i, i]];
        if pivot.abs() < 1e-10 {
            continue; // Skip near-zero pivots
        }
        for j in 0..(2 * n) {
            augmented[[i, j]] /= pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = augmented[[k, i]];
                for j in 0..(2 * n) {
                    augmented[[k, j]] -= factor * augmented[[i, j]];
                }
            }
        }
    }

    // Extract inverse
    augmented.slice(s![.., n..]).to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_esn_creation() {
        let config = ESNConfig::new(5, 1)
            .reservoir_size(100)
            .spectral_radius(0.9);

        let esn = EchoStateNetwork::new(config);
        assert_eq!(esn.config.reservoir_size, 100);
        assert_eq!(esn.config.input_dim, 5);
    }

    #[test]
    fn test_esn_update() {
        let config = ESNConfig::new(3, 1).reservoir_size(50);
        let mut esn = EchoStateNetwork::new(config);

        let input = Array1::from_vec(vec![0.5, -0.3, 0.8]);
        let state = esn.update(&input);

        assert_eq!(state.len(), 50);
    }

    #[test]
    fn test_spectral_radius() {
        let config = ESNConfig::new(1, 1)
            .reservoir_size(100)
            .spectral_radius(0.9)
            .seed(42);

        let esn = EchoStateNetwork::new(config);
        let radius = estimate_spectral_radius(&esn.w_res);

        // Should be close to 0.9 (with some tolerance)
        assert!(radius < 1.0);
        assert!(radius > 0.5);
    }

    #[test]
    fn test_ensemble() {
        let config = ESNConfig::new(3, 1).reservoir_size(50);
        let mut ensemble = EnsembleESN::new(config, 5);

        let input = Array1::from_vec(vec![0.5, -0.3, 0.8]);
        let (mean, variance) = ensemble.predict_with_uncertainty(&input);

        assert_eq!(mean.len(), 1);
        assert_eq!(variance.len(), 1);
    }
}
