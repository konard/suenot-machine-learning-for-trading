//! Reservoir dynamics and state management

use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for reservoir
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservoirConfig {
    /// Number of neurons
    pub size: usize,
    /// Sparsity (fraction of non-zero connections)
    pub sparsity: f64,
    /// Spectral radius
    pub spectral_radius: f64,
    /// Input dimension
    pub input_dim: usize,
    /// Input scaling
    pub input_scaling: f64,
    /// Leaking rate
    pub leaking_rate: f64,
}

impl Default for ReservoirConfig {
    fn default() -> Self {
        Self {
            size: 500,
            sparsity: 0.1,
            spectral_radius: 0.95,
            input_dim: 1,
            input_scaling: 0.1,
            leaking_rate: 0.3,
        }
    }
}

/// Reservoir component of ESN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reservoir {
    /// Configuration
    config: ReservoirConfig,
    /// Input weight matrix
    w_in: Array2<f64>,
    /// Recurrent weight matrix
    w: Array2<f64>,
    /// Current state
    state: Array1<f64>,
    /// Activation function type
    activation: ActivationType,
}

/// Types of activation functions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationType {
    Tanh,
    Sigmoid,
    ReLU,
    LeakyReLU(f64),
}

impl Reservoir {
    /// Create a new reservoir
    pub fn new(config: ReservoirConfig) -> Self {
        let mut rng = rand::thread_rng();

        let w_in = Self::create_input_weights(&config, &mut rng);
        let w = Self::create_recurrent_weights(&config, &mut rng);
        let state = Array1::zeros(config.size);

        Self {
            config,
            w_in,
            w,
            state,
            activation: ActivationType::Tanh,
        }
    }

    /// Create with specific random seed
    pub fn with_seed(config: ReservoirConfig, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let w_in = Self::create_input_weights(&config, &mut rng);
        let w = Self::create_recurrent_weights(&config, &mut rng);
        let state = Array1::zeros(config.size);

        Self {
            config,
            w_in,
            w,
            state,
            activation: ActivationType::Tanh,
        }
    }

    /// Set the activation function
    pub fn with_activation(mut self, activation: ActivationType) -> Self {
        self.activation = activation;
        self
    }

    fn create_input_weights<R: Rng>(config: &ReservoirConfig, rng: &mut R) -> Array2<f64> {
        let mut w = Array2::random_using(
            (config.size, config.input_dim),
            Uniform::new(-1.0, 1.0),
            rng,
        );
        w *= config.input_scaling;
        w
    }

    fn create_recurrent_weights<R: Rng>(config: &ReservoirConfig, rng: &mut R) -> Array2<f64> {
        let mut w = Array2::zeros((config.size, config.size));
        let dist = Uniform::new(-1.0, 1.0);

        // Create sparse connections
        for i in 0..config.size {
            for j in 0..config.size {
                if rng.gen::<f64>() < config.sparsity {
                    w[[i, j]] = rng.sample(dist);
                }
            }
        }

        // Scale to spectral radius
        let current_radius = Self::power_iteration_spectral_radius(&w, 50);
        if current_radius > 1e-10 {
            w *= config.spectral_radius / current_radius;
        }

        w
    }

    fn power_iteration_spectral_radius(matrix: &Array2<f64>, iterations: usize) -> f64 {
        let n = matrix.nrows();
        let mut v = Array1::random(n, Uniform::new(0.0, 1.0));
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v /= norm;

        for _ in 0..iterations {
            let v_new = matrix.dot(&v);
            let norm_new = v_new.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm_new < 1e-10 {
                return 0.0;
            }
            v = v_new / norm_new;
        }

        let av = matrix.dot(&v);
        let norm_av = av.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_v = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        norm_av / norm_v
    }

    /// Apply activation function
    fn activate(&self, x: f64) -> f64 {
        match self.activation {
            ActivationType::Tanh => x.tanh(),
            ActivationType::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationType::ReLU => x.max(0.0),
            ActivationType::LeakyReLU(alpha) => if x > 0.0 { x } else { alpha * x },
        }
    }

    /// Update reservoir state
    pub fn update(&mut self, input: &Array1<f64>) -> &Array1<f64> {
        // Pre-activation: W_in * input + W * state
        let pre_activation = self.w_in.dot(input) + self.w.dot(&self.state);

        // Apply activation and leaky integration
        let activated: Array1<f64> = pre_activation.mapv(|x| self.activate(x));

        self.state = &self.state * (1.0 - self.config.leaking_rate)
            + &activated * self.config.leaking_rate;

        &self.state
    }

    /// Reset state to zeros
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    /// Get current state
    pub fn state(&self) -> &Array1<f64> {
        &self.state
    }

    /// Get reservoir size
    pub fn size(&self) -> usize {
        self.config.size
    }

    /// Get spectral radius
    pub fn spectral_radius(&self) -> f64 {
        Self::power_iteration_spectral_radius(&self.w, 50)
    }

    /// Compute memory capacity (diagnostic)
    pub fn memory_capacity(&self, test_length: usize) -> f64 {
        // Generate random test signal
        let mut rng = rand::thread_rng();
        let signal: Vec<f64> = (0..test_length)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        // Run through reservoir and collect states
        self.reset();
        let mut states = Vec::new();
        for &s in &signal {
            let input = Array1::from_vec(vec![s]);
            self.update(&input);
            states.push(self.state.clone());
        }

        // Compute memory capacity for different delays
        let mut total_mc = 0.0;
        let max_delay = test_length / 2;

        for delay in 1..max_delay {
            let mut corr_sum = 0.0;
            let mut count = 0;

            for t in delay..test_length {
                // Correlation between state[t] and signal[t-delay]
                let state_mean: f64 = states[t].mean().unwrap_or(0.0);
                corr_sum += (states[t][0] - state_mean) * signal[t - delay];
                count += 1;
            }

            if count > 0 {
                let corr = corr_sum / count as f64;
                total_mc += corr * corr;
            }
        }

        total_mc
    }
}

/// Intrinsic Plasticity for adapting reservoir neurons
pub struct IntrinsicPlasticity {
    /// Target mean activation
    target_mean: f64,
    /// Target variance
    target_var: f64,
    /// Learning rate
    learning_rate: f64,
    /// Gain parameters
    gains: Array1<f64>,
    /// Bias parameters
    biases: Array1<f64>,
}

impl IntrinsicPlasticity {
    /// Create new IP adapter
    pub fn new(size: usize) -> Self {
        Self {
            target_mean: 0.0,
            target_var: 0.1,
            learning_rate: 0.001,
            gains: Array1::ones(size),
            biases: Array1::zeros(size),
        }
    }

    /// Adapt based on current activation
    pub fn adapt(&mut self, activation: &Array1<f64>) {
        for i in 0..activation.len() {
            let y = activation[i];

            // Update bias
            let delta_b = self.learning_rate * (self.target_mean - y);
            self.biases[i] += delta_b;

            // Update gain
            let y2 = y * y;
            let delta_a = self.learning_rate * (1.0 / self.gains[i] + y * (2.0 * self.target_var - y2));
            self.gains[i] += delta_a;
            self.gains[i] = self.gains[i].max(0.01); // Prevent negative gains
        }
    }

    /// Transform activation
    pub fn transform(&self, pre_activation: &Array1<f64>) -> Array1<f64> {
        (&self.gains * pre_activation + &self.biases).mapv(|x| x.tanh())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reservoir_creation() {
        let config = ReservoirConfig {
            size: 100,
            sparsity: 0.1,
            spectral_radius: 0.9,
            input_dim: 5,
            input_scaling: 0.1,
            leaking_rate: 0.3,
        };

        let reservoir = Reservoir::new(config);
        assert_eq!(reservoir.size(), 100);
    }

    #[test]
    fn test_reservoir_update() {
        let config = ReservoirConfig {
            size: 50,
            input_dim: 3,
            ..Default::default()
        };

        let mut reservoir = Reservoir::new(config);
        let input = Array1::from_vec(vec![0.5, -0.3, 0.8]);

        let state = reservoir.update(&input);
        assert_eq!(state.len(), 50);

        // State should be non-zero after update
        assert!(state.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_spectral_radius_scaling() {
        let config = ReservoirConfig {
            size: 100,
            spectral_radius: 0.9,
            ..Default::default()
        };

        let reservoir = Reservoir::with_seed(config, 42);
        let actual_radius = reservoir.spectral_radius();

        // Should be close to target (with tolerance)
        assert!(actual_radius < 1.0);
        assert!(actual_radius > 0.5);
    }

    #[test]
    fn test_activation_types() {
        let config = ReservoirConfig::default();

        let tanh_res = Reservoir::new(config.clone())
            .with_activation(ActivationType::Tanh);
        let relu_res = Reservoir::new(config.clone())
            .with_activation(ActivationType::ReLU);
        let sigmoid_res = Reservoir::new(config)
            .with_activation(ActivationType::Sigmoid);

        assert!(matches!(tanh_res.activation, ActivationType::Tanh));
        assert!(matches!(relu_res.activation, ActivationType::ReLU));
        assert!(matches!(sigmoid_res.activation, ActivationType::Sigmoid));
    }
}
