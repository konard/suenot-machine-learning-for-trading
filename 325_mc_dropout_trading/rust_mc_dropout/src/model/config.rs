//! Configuration for MC Dropout models

use serde::{Deserialize, Serialize};

/// Configuration for MC Dropout model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCDropoutConfig {
    /// Number of input features
    pub input_dim: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Output dimension
    pub output_dim: usize,
    /// Dropout probability
    pub dropout_rate: f64,
    /// Number of MC samples for inference
    pub n_mc_samples: usize,
    /// Learning rate for training
    pub learning_rate: f64,
    /// L2 regularization weight
    pub weight_decay: f64,
}

impl Default for MCDropoutConfig {
    fn default() -> Self {
        Self {
            input_dim: 20,
            hidden_dims: vec![128, 64, 32],
            output_dim: 1,
            dropout_rate: 0.1,
            n_mc_samples: 50,
            learning_rate: 0.001,
            weight_decay: 0.01,
        }
    }
}

impl MCDropoutConfig {
    /// Create a new configuration
    pub fn new(input_dim: usize) -> Self {
        Self {
            input_dim,
            ..Default::default()
        }
    }

    /// Set hidden layer dimensions
    pub fn with_hidden_dims(mut self, dims: Vec<usize>) -> Self {
        self.hidden_dims = dims;
        self
    }

    /// Set dropout rate
    pub fn with_dropout_rate(mut self, rate: f64) -> Self {
        self.dropout_rate = rate.clamp(0.0, 0.9);
        self
    }

    /// Set number of MC samples
    pub fn with_mc_samples(mut self, n: usize) -> Self {
        self.n_mc_samples = n.max(1);
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.input_dim == 0 {
            return Err("Input dimension must be positive".to_string());
        }
        if self.hidden_dims.is_empty() {
            return Err("Must have at least one hidden layer".to_string());
        }
        if self.dropout_rate < 0.0 || self.dropout_rate >= 1.0 {
            return Err("Dropout rate must be in [0, 1)".to_string());
        }
        if self.n_mc_samples == 0 {
            return Err("Number of MC samples must be positive".to_string());
        }
        Ok(())
    }
}
