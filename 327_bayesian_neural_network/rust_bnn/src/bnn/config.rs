//! Configuration for Bayesian Neural Networks.

use serde::{Deserialize, Serialize};

/// Configuration for Bayesian Neural Network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BNNConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Number of output classes
    pub num_classes: usize,
    /// First prior standard deviation (for mixture)
    pub prior_sigma_1: f64,
    /// Second prior standard deviation (for mixture)
    pub prior_sigma_2: f64,
    /// Mixture coefficient for scale mixture prior
    pub prior_pi: f64,
    /// Dropout rate
    pub dropout_rate: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of MC samples for training
    pub n_samples_train: usize,
    /// Number of MC samples for inference
    pub n_samples_inference: usize,
    /// KL weight (beta for annealing)
    pub kl_weight: f64,
    /// Whether to use heteroscedastic output
    pub heteroscedastic: bool,
}

impl Default for BNNConfig {
    fn default() -> Self {
        Self {
            input_dim: 20,
            hidden_dims: vec![128, 64, 32],
            num_classes: 3,
            prior_sigma_1: 1.0,
            prior_sigma_2: 0.0025,
            prior_pi: 0.5,
            dropout_rate: 0.1,
            learning_rate: 0.001,
            n_samples_train: 3,
            n_samples_inference: 100,
            kl_weight: 1.0,
            heteroscedastic: true,
        }
    }
}

impl BNNConfig {
    /// Create a new configuration with specified input dimension.
    pub fn with_input_dim(input_dim: usize) -> Self {
        Self {
            input_dim,
            ..Default::default()
        }
    }

    /// Set hidden layer dimensions.
    pub fn hidden_dims(mut self, dims: Vec<usize>) -> Self {
        self.hidden_dims = dims;
        self
    }

    /// Set number of output classes.
    pub fn num_classes(mut self, n: usize) -> Self {
        self.num_classes = n;
        self
    }

    /// Set prior parameters.
    pub fn prior(mut self, sigma_1: f64, sigma_2: f64, pi: f64) -> Self {
        self.prior_sigma_1 = sigma_1;
        self.prior_sigma_2 = sigma_2;
        self.prior_pi = pi;
        self
    }

    /// Set learning rate.
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set MC samples for inference.
    pub fn n_samples_inference(mut self, n: usize) -> Self {
        self.n_samples_inference = n;
        self
    }

    /// Calculate total number of parameters.
    pub fn total_parameters(&self) -> usize {
        let mut total = 0;
        let mut prev_dim = self.input_dim;

        for &hidden_dim in &self.hidden_dims {
            // Each weight has 2 parameters (mu and rho)
            total += 2 * prev_dim * hidden_dim;
            // Bias
            total += 2 * hidden_dim;
            prev_dim = hidden_dim;
        }

        // Output layer
        total += 2 * prev_dim * self.num_classes;
        total += 2 * self.num_classes;

        // Heteroscedastic head
        if self.heteroscedastic {
            total += 2 * prev_dim * 2;
            total += 2 * 2;
        }

        total
    }
}

/// KL annealing schedule.
#[derive(Debug, Clone, Copy)]
pub enum KLSchedule {
    /// Linear warmup
    Linear { warmup_epochs: usize },
    /// Sigmoid warmup
    Sigmoid { warmup_epochs: usize },
    /// Cyclical annealing
    Cyclical { cycle_length: usize },
    /// Constant (no annealing)
    Constant,
}

impl KLSchedule {
    /// Get KL weight for a given epoch.
    pub fn get_weight(&self, epoch: usize) -> f64 {
        match self {
            KLSchedule::Linear { warmup_epochs } => {
                if epoch >= *warmup_epochs {
                    1.0
                } else {
                    (epoch + 1) as f64 / *warmup_epochs as f64
                }
            }
            KLSchedule::Sigmoid { warmup_epochs } => {
                if epoch >= *warmup_epochs {
                    1.0
                } else {
                    let x = 10.0 * (epoch as f64 / *warmup_epochs as f64 - 0.5);
                    1.0 / (1.0 + (-x).exp())
                }
            }
            KLSchedule::Cyclical { cycle_length } => {
                let position = epoch % cycle_length;
                position as f64 / *cycle_length as f64
            }
            KLSchedule::Constant => 1.0,
        }
    }
}
