//! # Reservoir Computing Core
//!
//! Implementation of Echo State Networks (ESN) for time series prediction.
//!
//! ## Overview
//!
//! The Echo State Network consists of:
//! - Input layer: Maps input features to reservoir
//! - Reservoir: Fixed, randomly connected recurrent layer
//! - Output layer: Trained linear readout
//!
//! ## Example
//!
//! ```rust
//! use reservoir_trading::reservoir::{EchoStateNetwork, EsnConfig};
//!
//! let config = EsnConfig {
//!     reservoir_size: 500,
//!     spectral_radius: 0.95,
//!     input_scaling: 0.5,
//!     leaking_rate: 0.3,
//!     sparsity: 0.1,
//!     regularization: 1e-6,
//!     seed: 42,
//! };
//!
//! let mut esn = EchoStateNetwork::new(7, 1, config);
//! ```

use ndarray::{Array1, Array2, s, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::{SeedableRng, Rng};
use rand::rngs::StdRng;
use thiserror::Error;

/// Errors that can occur in reservoir operations
#[derive(Error, Debug)]
pub enum ReservoirError {
    #[error("ESN not trained yet")]
    NotTrained,

    #[error("Invalid input dimension: expected {expected}, got {got}")]
    InvalidInputDimension { expected: usize, got: usize },

    #[error("Empty input data")]
    EmptyInput,

    #[error("Washout period ({washout}) larger than data length ({data_len})")]
    WashoutTooLarge { washout: usize, data_len: usize },

    #[error("Linear algebra error: {0}")]
    LinalgError(String),
}

/// Configuration for Echo State Network
#[derive(Debug, Clone)]
pub struct EsnConfig {
    /// Number of neurons in the reservoir
    pub reservoir_size: usize,

    /// Spectral radius of reservoir matrix (controls memory length)
    pub spectral_radius: f64,

    /// Scaling factor for input weights
    pub input_scaling: f64,

    /// Leaking rate (controls temporal smoothing)
    pub leaking_rate: f64,

    /// Sparsity of reservoir connections (fraction of non-zero weights)
    pub sparsity: f64,

    /// Ridge regression regularization parameter
    pub regularization: f64,

    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for EsnConfig {
    fn default() -> Self {
        Self {
            reservoir_size: 500,
            spectral_radius: 0.95,
            input_scaling: 0.5,
            leaking_rate: 0.3,
            sparsity: 0.1,
            regularization: 1e-6,
            seed: 42,
        }
    }
}

impl EsnConfig {
    /// Create a new configuration for high-frequency trading
    pub fn hft() -> Self {
        Self {
            reservoir_size: 200,
            spectral_radius: 0.8,
            input_scaling: 0.3,
            leaking_rate: 0.5,
            sparsity: 0.15,
            regularization: 1e-5,
            seed: 42,
        }
    }

    /// Create a new configuration for swing trading
    pub fn swing() -> Self {
        Self {
            reservoir_size: 1000,
            spectral_radius: 0.99,
            input_scaling: 0.5,
            leaking_rate: 0.2,
            sparsity: 0.05,
            regularization: 1e-7,
            seed: 42,
        }
    }
}

/// Echo State Network for time series prediction
pub struct EchoStateNetwork {
    /// Number of input features
    n_inputs: usize,

    /// Number of output features
    n_outputs: usize,

    /// Configuration parameters
    config: EsnConfig,

    /// Input weight matrix (reservoir_size x n_inputs)
    w_in: Array2<f64>,

    /// Reservoir weight matrix (reservoir_size x reservoir_size)
    w_res: Array2<f64>,

    /// Output weight matrix (n_outputs x (1 + n_inputs + reservoir_size))
    w_out: Option<Array2<f64>>,

    /// Current reservoir state
    state: Array1<f64>,

    /// Whether the network has been trained
    trained: bool,
}

impl EchoStateNetwork {
    /// Create a new Echo State Network
    ///
    /// # Arguments
    ///
    /// * `n_inputs` - Number of input features
    /// * `n_outputs` - Number of output features
    /// * `config` - ESN configuration
    ///
    /// # Example
    ///
    /// ```rust
    /// use reservoir_trading::reservoir::{EchoStateNetwork, EsnConfig};
    ///
    /// let esn = EchoStateNetwork::new(7, 1, EsnConfig::default());
    /// ```
    pub fn new(n_inputs: usize, n_outputs: usize, config: EsnConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);

        // Initialize input weights: uniform [-1, 1] scaled
        let w_in = Array2::random_using(
            (config.reservoir_size, n_inputs),
            Uniform::new(-1.0, 1.0),
            &mut rng,
        ) * config.input_scaling;

        // Initialize reservoir weights: sparse random matrix
        let mut w_res = Array2::random_using(
            (config.reservoir_size, config.reservoir_size),
            Uniform::new(-1.0, 1.0),
            &mut rng,
        );

        // Apply sparsity mask
        for elem in w_res.iter_mut() {
            if rng.gen::<f64>() > config.sparsity {
                *elem = 0.0;
            }
        }

        // Scale to desired spectral radius
        let spectral_radius = estimate_spectral_radius(&w_res);
        if spectral_radius > 0.0 {
            w_res *= config.spectral_radius / spectral_radius;
        }

        // Initialize state to zeros
        let state = Array1::zeros(config.reservoir_size);

        Self {
            n_inputs,
            n_outputs,
            config,
            w_in,
            w_res,
            w_out: None,
            state,
            trained: false,
        }
    }

    /// Reset the reservoir state to zeros
    pub fn reset_state(&mut self) {
        self.state = Array1::zeros(self.config.reservoir_size);
    }

    /// Get the current reservoir state
    pub fn state(&self) -> &Array1<f64> {
        &self.state
    }

    /// Update the reservoir state with a single input
    fn update_state(&mut self, input: &Array1<f64>) {
        let pre_activation = self.w_in.dot(input) + self.w_res.dot(&self.state);

        // Apply leaking rate and tanh activation
        self.state = &self.state * (1.0 - self.config.leaking_rate)
            + pre_activation.mapv(|x| x.tanh()) * self.config.leaking_rate;
    }

    /// Collect reservoir states for a sequence of inputs
    fn collect_states(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
        let n_samples = inputs.nrows();
        let mut states = Array2::zeros((n_samples, self.config.reservoir_size));

        for t in 0..n_samples {
            let input = inputs.row(t).to_owned();
            self.update_state(&input);
            states.row_mut(t).assign(&self.state);
        }

        states
    }

    /// Train the ESN using ridge regression
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input sequences, shape (n_samples, n_inputs)
    /// * `targets` - Target outputs, shape (n_samples, n_outputs)
    /// * `washout` - Initial transient period to discard
    ///
    /// # Returns
    ///
    /// Training error (MSE)
    pub fn fit(
        &mut self,
        inputs: &Array2<f64>,
        targets: &Array2<f64>,
        washout: usize,
    ) -> Result<f64, ReservoirError> {
        let n_samples = inputs.nrows();

        if n_samples == 0 {
            return Err(ReservoirError::EmptyInput);
        }

        if inputs.ncols() != self.n_inputs {
            return Err(ReservoirError::InvalidInputDimension {
                expected: self.n_inputs,
                got: inputs.ncols(),
            });
        }

        if washout >= n_samples {
            return Err(ReservoirError::WashoutTooLarge {
                washout,
                data_len: n_samples,
            });
        }

        // Reset state and collect states
        self.reset_state();
        let states = self.collect_states(inputs);

        // Discard washout period
        let states = states.slice(s![washout.., ..]).to_owned();
        let inputs_trimmed = inputs.slice(s![washout.., ..]).to_owned();
        let targets_trimmed = targets.slice(s![washout.., ..]).to_owned();

        let n_effective = n_samples - washout;

        // Construct extended state matrix [1, input, state]
        let extended_size = 1 + self.n_inputs + self.config.reservoir_size;
        let mut extended_states = Array2::zeros((n_effective, extended_size));

        // Add bias column
        extended_states.column_mut(0).fill(1.0);

        // Add input columns
        extended_states
            .slice_mut(s![.., 1..=self.n_inputs])
            .assign(&inputs_trimmed);

        // Add state columns
        extended_states
            .slice_mut(s![.., (1 + self.n_inputs)..])
            .assign(&states);

        // Ridge regression: W_out = (S^T S + λI)^(-1) S^T y
        let s_t = extended_states.t();
        let s_t_s = s_t.dot(&extended_states);

        // Add regularization
        let mut reg_matrix = Array2::eye(extended_size) * self.config.regularization;
        let s_t_s_reg = &s_t_s + &reg_matrix;

        // Solve using Cholesky decomposition (more stable than direct inverse)
        let s_t_y = s_t.dot(&targets_trimmed);

        // Simple solve using pseudo-inverse approach
        let w_out = solve_ridge(&s_t_s_reg, &s_t_y)?;

        self.w_out = Some(w_out.t().to_owned());
        self.trained = true;

        // Calculate training error
        let predictions = extended_states.dot(&self.w_out.as_ref().unwrap().t());
        let errors = &predictions - &targets_trimmed;
        let mse = errors.mapv(|x| x * x).mean().unwrap_or(0.0);

        Ok(mse)
    }

    /// Make predictions for a sequence of inputs
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input sequences, shape (n_samples, n_inputs)
    ///
    /// # Returns
    ///
    /// Predictions, shape (n_samples, n_outputs)
    pub fn predict(&mut self, inputs: &Array2<f64>) -> Result<Array2<f64>, ReservoirError> {
        if !self.trained {
            return Err(ReservoirError::NotTrained);
        }

        if inputs.ncols() != self.n_inputs {
            return Err(ReservoirError::InvalidInputDimension {
                expected: self.n_inputs,
                got: inputs.ncols(),
            });
        }

        let n_samples = inputs.nrows();
        let states = self.collect_states(inputs);

        // Construct extended states
        let extended_size = 1 + self.n_inputs + self.config.reservoir_size;
        let mut extended_states = Array2::zeros((n_samples, extended_size));

        extended_states.column_mut(0).fill(1.0);
        extended_states
            .slice_mut(s![.., 1..=self.n_inputs])
            .assign(inputs);
        extended_states
            .slice_mut(s![.., (1 + self.n_inputs)..])
            .assign(&states);

        // Compute predictions
        let w_out = self.w_out.as_ref().unwrap();
        let predictions = extended_states.dot(&w_out.t());

        Ok(predictions)
    }

    /// Make a single prediction and update state
    pub fn predict_one(&mut self, input: &Array1<f64>) -> Result<Array1<f64>, ReservoirError> {
        if !self.trained {
            return Err(ReservoirError::NotTrained);
        }

        if input.len() != self.n_inputs {
            return Err(ReservoirError::InvalidInputDimension {
                expected: self.n_inputs,
                got: input.len(),
            });
        }

        // Update state
        self.update_state(input);

        // Construct extended state
        let extended_size = 1 + self.n_inputs + self.config.reservoir_size;
        let mut extended_state = Array1::zeros(extended_size);
        extended_state[0] = 1.0;
        extended_state.slice_mut(s![1..=self.n_inputs]).assign(input);
        extended_state.slice_mut(s![(1 + self.n_inputs)..]).assign(&self.state);

        // Compute prediction
        let w_out = self.w_out.as_ref().unwrap();
        let prediction = w_out.dot(&extended_state);

        Ok(prediction)
    }

    /// Get the number of trainable parameters
    pub fn n_parameters(&self) -> usize {
        self.n_outputs * (1 + self.n_inputs + self.config.reservoir_size)
    }

    /// Check if the network is trained
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Get the configuration
    pub fn config(&self) -> &EsnConfig {
        &self.config
    }
}

/// Online ESN with Recursive Least Squares (RLS) training
pub struct OnlineEsn {
    /// Base ESN
    esn: EchoStateNetwork,

    /// Forgetting factor for RLS
    forgetting_factor: f64,

    /// Inverse covariance matrix
    p_matrix: Option<Array2<f64>>,

    /// Output weights as mutable array
    w_out: Option<Array2<f64>>,
}

impl OnlineEsn {
    /// Create a new Online ESN
    pub fn new(n_inputs: usize, n_outputs: usize, config: EsnConfig, forgetting_factor: f64) -> Self {
        let esn = EchoStateNetwork::new(n_inputs, n_outputs, config);

        Self {
            esn,
            forgetting_factor,
            p_matrix: None,
            w_out: None,
        }
    }

    /// Initialize the RLS matrices
    fn initialize_rls(&mut self) {
        let extended_size = 1 + self.esn.n_inputs + self.esn.config.reservoir_size;

        // Initialize P matrix (inverse covariance)
        self.p_matrix = Some(Array2::eye(extended_size) / self.esn.config.regularization);

        // Initialize output weights to zeros
        self.w_out = Some(Array2::zeros((self.esn.n_outputs, extended_size)));
    }

    /// Perform one step of online learning
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector
    /// * `target` - Target output vector
    ///
    /// # Returns
    ///
    /// Prediction made before the update
    pub fn partial_fit(&mut self, input: &Array1<f64>, target: &Array1<f64>) -> Array1<f64> {
        // Initialize if needed
        if self.p_matrix.is_none() {
            self.initialize_rls();
        }

        // Update reservoir state
        self.esn.update_state(input);

        // Construct extended state vector
        let extended_size = 1 + self.esn.n_inputs + self.esn.config.reservoir_size;
        let mut phi = Array1::zeros(extended_size);
        phi[0] = 1.0;
        phi.slice_mut(s![1..=self.esn.n_inputs]).assign(input);
        phi.slice_mut(s![(1 + self.esn.n_inputs)..]).assign(&self.esn.state);

        let p = self.p_matrix.as_ref().unwrap();
        let w_out = self.w_out.as_mut().unwrap();

        // RLS update
        let lambda = self.forgetting_factor;

        // k = P * phi / (lambda + phi^T * P * phi)
        let p_phi = p.dot(&phi);
        let denominator = lambda + phi.dot(&p_phi);
        let k = &p_phi / denominator;

        // Prediction before update
        let prediction = w_out.dot(&phi);

        // Error
        let error = target - &prediction;

        // Update weights: W_out = W_out + outer(error, k)
        for i in 0..self.esn.n_outputs {
            for j in 0..extended_size {
                w_out[[i, j]] += error[i] * k[j];
            }
        }

        // Update P matrix: P = (P - outer(k, phi^T * P)) / lambda
        let phi_t_p = phi.dot(p);
        let k_phi_t_p = outer_product(&k, &phi_t_p);
        let new_p = (p - &k_phi_t_p) / lambda;
        self.p_matrix = Some(new_p);

        prediction
    }

    /// Make a prediction without updating
    pub fn predict_one(&mut self, input: &Array1<f64>) -> Option<Array1<f64>> {
        if self.w_out.is_none() {
            return None;
        }

        // Update state
        self.esn.update_state(input);

        // Construct extended state
        let extended_size = 1 + self.esn.n_inputs + self.esn.config.reservoir_size;
        let mut phi = Array1::zeros(extended_size);
        phi[0] = 1.0;
        phi.slice_mut(s![1..=self.esn.n_inputs]).assign(input);
        phi.slice_mut(s![(1 + self.esn.n_inputs)..]).assign(&self.esn.state);

        let w_out = self.w_out.as_ref().unwrap();
        Some(w_out.dot(&phi))
    }

    /// Reset the network state
    pub fn reset(&mut self) {
        self.esn.reset_state();
    }

    /// Get the forgetting factor
    pub fn forgetting_factor(&self) -> f64 {
        self.forgetting_factor
    }
}

/// Estimate spectral radius using power iteration
fn estimate_spectral_radius(matrix: &Array2<f64>) -> f64 {
    let n = matrix.nrows();
    let mut v = Array1::from_elem(n, 1.0 / (n as f64).sqrt());

    // Power iteration
    for _ in 0..100 {
        let v_new = matrix.dot(&v);
        let norm = v_new.mapv(|x| x * x).sum().sqrt();
        if norm < 1e-10 {
            return 0.0;
        }
        v = v_new / norm;
    }

    // Rayleigh quotient
    let av = matrix.dot(&v);
    v.dot(&av) / v.dot(&v)
}

/// Solve ridge regression using Cholesky-like decomposition
fn solve_ridge(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, ReservoirError> {
    // Simple pseudo-inverse approach for small matrices
    // For production, use ndarray-linalg with LAPACK

    let n = a.nrows();
    let m = b.ncols();

    // Use Gauss-Jordan elimination
    let mut aug = Array2::zeros((n, n + m));
    aug.slice_mut(s![.., ..n]).assign(a);
    aug.slice_mut(s![.., n..]).assign(b);

    // Forward elimination with partial pivoting
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        let mut max_val = aug[[i, i]].abs();
        for k in (i + 1)..n {
            if aug[[k, i]].abs() > max_val {
                max_val = aug[[k, i]].abs();
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            for j in 0..(n + m) {
                let tmp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        // Check for singularity
        if aug[[i, i]].abs() < 1e-12 {
            return Err(ReservoirError::LinalgError(
                "Matrix is singular or nearly singular".to_string()
            ));
        }

        // Eliminate column
        for k in (i + 1)..n {
            let factor = aug[[k, i]] / aug[[i, i]];
            for j in i..(n + m) {
                aug[[k, j]] -= factor * aug[[i, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array2::zeros((n, m));
    for i in (0..n).rev() {
        for j in 0..m {
            let mut sum = aug[[i, n + j]];
            for k in (i + 1)..n {
                sum -= aug[[i, k]] * x[[k, j]];
            }
            x[[i, j]] = sum / aug[[i, i]];
        }
    }

    Ok(x)
}

/// Compute outer product of two vectors
fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let m = b.len();
    let mut result = Array2::zeros((n, m));

    for i in 0..n {
        for j in 0..m {
            result[[i, j]] = a[i] * b[j];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_esn_creation() {
        let esn = EchoStateNetwork::new(5, 1, EsnConfig::default());
        assert_eq!(esn.n_inputs, 5);
        assert_eq!(esn.n_outputs, 1);
        assert!(!esn.is_trained());
    }

    #[test]
    fn test_esn_state_update() {
        let mut esn = EchoStateNetwork::new(3, 1, EsnConfig::default());
        let input = Array1::from_vec(vec![0.5, -0.3, 0.1]);

        esn.update_state(&input);

        // State should be non-zero after update
        assert!(esn.state.iter().any(|&x| x.abs() > 1e-10));
    }

    #[test]
    fn test_esn_train_predict() {
        let mut esn = EchoStateNetwork::new(2, 1, EsnConfig {
            reservoir_size: 50,
            ..EsnConfig::default()
        });

        // Generate simple training data
        let n_samples = 200;
        let mut inputs = Array2::zeros((n_samples, 2));
        let mut targets = Array2::zeros((n_samples, 1));

        for i in 0..n_samples {
            let t = i as f64 / 10.0;
            inputs[[i, 0]] = t.sin();
            inputs[[i, 1]] = t.cos();
            targets[[i, 0]] = (t + 0.1).sin(); // Predict future
        }

        // Train
        let mse = esn.fit(&inputs, &targets, 50).unwrap();
        assert!(esn.is_trained());

        // Predict
        esn.reset_state();
        let predictions = esn.predict(&inputs).unwrap();
        assert_eq!(predictions.nrows(), n_samples);
    }

    #[test]
    fn test_online_esn() {
        let config = EsnConfig {
            reservoir_size: 50,
            ..EsnConfig::default()
        };
        let mut online_esn = OnlineEsn::new(2, 1, config, 0.995);

        // Online training
        for i in 0..100 {
            let t = i as f64 / 10.0;
            let input = Array1::from_vec(vec![t.sin(), t.cos()]);
            let target = Array1::from_vec(vec![(t + 0.1).sin()]);

            let _ = online_esn.partial_fit(&input, &target);
        }

        // Should be able to predict
        let test_input = Array1::from_vec(vec![0.5, 0.866]);
        let prediction = online_esn.predict_one(&test_input);
        assert!(prediction.is_some());
    }

    #[test]
    fn test_spectral_radius() {
        let mut rng = StdRng::seed_from_u64(42);
        let matrix = Array2::random_using((10, 10), Uniform::new(-1.0, 1.0), &mut rng);

        let rho = estimate_spectral_radius(&matrix);
        // Spectral radius should be positive for random matrix
        assert!(rho > 0.0);
    }
}
