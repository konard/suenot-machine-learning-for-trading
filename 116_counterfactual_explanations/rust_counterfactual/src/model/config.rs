//! Model configuration.

use serde::{Deserialize, Serialize};

/// Configuration for trading classifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Number of input features
    pub input_dim: usize,
    /// Hidden layer dimension
    pub hidden_dim: usize,
    /// Number of output classes
    pub num_classes: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Dropout probability
    pub dropout: f64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            input_dim: 6,
            hidden_dim: 64,
            num_classes: 3,
            learning_rate: 0.01,
            dropout: 0.2,
        }
    }
}

impl ModelConfig {
    /// Create a new configuration.
    pub fn new(input_dim: usize, hidden_dim: usize, num_classes: usize) -> Self {
        Self {
            input_dim,
            hidden_dim,
            num_classes,
            ..Default::default()
        }
    }

    /// Set learning rate.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set dropout probability.
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }
}
