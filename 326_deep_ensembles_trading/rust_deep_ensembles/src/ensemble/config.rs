//! Configuration for Deep Ensemble models

use serde::{Deserialize, Serialize};

/// Configuration for Deep Ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Number of ensemble members
    pub num_models: usize,
    /// Input dimension
    pub input_dim: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Dropout rate
    pub dropout_rate: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size for training
    pub batch_size: usize,
    /// Maximum training epochs
    pub max_epochs: usize,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Minimum standard deviation (prevents numerical issues)
    pub min_std: f64,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            num_models: 5,
            input_dim: 8,
            hidden_dims: vec![128, 64, 32],
            dropout_rate: 0.1,
            learning_rate: 0.001,
            batch_size: 64,
            max_epochs: 100,
            early_stopping_patience: 10,
            min_std: 1e-6,
        }
    }
}

impl EnsembleConfig {
    /// Create a minimal configuration for testing
    pub fn minimal() -> Self {
        Self {
            num_models: 3,
            input_dim: 4,
            hidden_dims: vec![32, 16],
            dropout_rate: 0.1,
            learning_rate: 0.01,
            batch_size: 32,
            max_epochs: 50,
            early_stopping_patience: 5,
            min_std: 1e-6,
        }
    }

    /// Create a large configuration for production
    pub fn large() -> Self {
        Self {
            num_models: 10,
            input_dim: 16,
            hidden_dims: vec![256, 128, 64],
            dropout_rate: 0.2,
            learning_rate: 0.0005,
            batch_size: 128,
            max_epochs: 200,
            early_stopping_patience: 20,
            min_std: 1e-6,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.num_models == 0 {
            return Err("num_models must be > 0".to_string());
        }
        if self.input_dim == 0 {
            return Err("input_dim must be > 0".to_string());
        }
        if self.hidden_dims.is_empty() {
            return Err("hidden_dims must not be empty".to_string());
        }
        if self.learning_rate <= 0.0 {
            return Err("learning_rate must be > 0".to_string());
        }
        if self.batch_size == 0 {
            return Err("batch_size must be > 0".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EnsembleConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.num_models, 5);
    }

    #[test]
    fn test_minimal_config() {
        let config = EnsembleConfig::minimal();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_config() {
        let mut config = EnsembleConfig::default();
        config.num_models = 0;
        assert!(config.validate().is_err());
    }
}
