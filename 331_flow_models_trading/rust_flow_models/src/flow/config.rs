//! Flow model configuration

use serde::{Deserialize, Serialize};

/// Configuration for normalizing flow model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Number of flow layers
    pub num_layers: usize,
    /// Hidden dimension for coupling networks
    pub hidden_dim: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Weight decay for regularization
    pub weight_decay: f64,
    /// Batch size for training
    pub batch_size: usize,
    /// Maximum epochs
    pub max_epochs: usize,
    /// Early stopping patience
    pub patience: usize,
    /// Gradient clipping threshold
    pub grad_clip: f64,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            input_dim: 8,
            num_layers: 6,
            hidden_dim: 128,
            learning_rate: 1e-4,
            weight_decay: 1e-5,
            batch_size: 128,
            max_epochs: 100,
            patience: 10,
            grad_clip: 1.0,
        }
    }
}

impl FlowConfig {
    /// Create config with custom input dimension
    pub fn with_input_dim(mut self, dim: usize) -> Self {
        self.input_dim = dim;
        self
    }

    /// Create config with custom number of layers
    pub fn with_num_layers(mut self, layers: usize) -> Self {
        self.num_layers = layers;
        self
    }

    /// Create config with custom hidden dimension
    pub fn with_hidden_dim(mut self, dim: usize) -> Self {
        self.hidden_dim = dim;
        self
    }

    /// Create config with custom learning rate
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.input_dim == 0 {
            return Err("input_dim must be > 0".to_string());
        }
        if self.num_layers == 0 {
            return Err("num_layers must be > 0".to_string());
        }
        if self.hidden_dim == 0 {
            return Err("hidden_dim must be > 0".to_string());
        }
        if self.learning_rate <= 0.0 {
            return Err("learning_rate must be > 0".to_string());
        }
        Ok(())
    }
}
