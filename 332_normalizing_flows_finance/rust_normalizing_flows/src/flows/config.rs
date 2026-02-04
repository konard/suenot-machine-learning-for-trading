//! Configuration for normalizing flow models.

use serde::{Deserialize, Serialize};

/// Configuration for normalizing flow models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowConfig {
    /// Input dimension
    pub dim: usize,
    /// Number of flow layers
    pub n_layers: usize,
    /// Hidden layer dimension
    pub hidden_dim: usize,
    /// Number of hidden layers in coupling networks
    pub n_hidden_layers: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Whether to use batch normalization
    pub use_batch_norm: bool,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            dim: 1,
            n_layers: 8,
            hidden_dim: 128,
            n_hidden_layers: 2,
            learning_rate: 0.001,
            use_batch_norm: false,
        }
    }
}

impl FlowConfig {
    /// Create a new configuration with specified dimension
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            ..Default::default()
        }
    }

    /// Builder pattern: set number of layers
    pub fn with_n_layers(mut self, n_layers: usize) -> Self {
        self.n_layers = n_layers;
        self
    }

    /// Builder pattern: set hidden dimension
    pub fn with_hidden_dim(mut self, hidden_dim: usize) -> Self {
        self.hidden_dim = hidden_dim;
        self
    }

    /// Builder pattern: set learning rate
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Builder pattern: enable batch normalization
    pub fn with_batch_norm(mut self, use_bn: bool) -> Self {
        self.use_batch_norm = use_bn;
        self
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Maximum number of epochs
    pub max_epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Early stopping patience
    pub patience: usize,
    /// Validation split ratio
    pub val_split: f64,
    /// Gradient clipping value
    pub grad_clip: f64,
    /// Whether to shuffle data each epoch
    pub shuffle: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_epochs: 200,
            batch_size: 64,
            patience: 20,
            val_split: 0.15,
            grad_clip: 1.0,
            shuffle: true,
        }
    }
}
