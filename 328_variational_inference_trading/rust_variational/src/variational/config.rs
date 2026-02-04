//! VAE configuration

use serde::{Deserialize, Serialize};

/// Configuration for VAE model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VAEConfig {
    /// Input feature dimension
    pub input_dim: usize,

    /// Latent space dimension
    pub latent_dim: usize,

    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,

    /// Beta coefficient for KL divergence
    pub beta: f64,

    /// Dropout rate
    pub dropout: f64,

    /// Sequence length
    pub sequence_length: usize,

    /// Number of market regimes to detect
    pub n_regimes: usize,

    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for VAEConfig {
    fn default() -> Self {
        Self {
            input_dim: 13,
            latent_dim: 16,
            hidden_dims: vec![256, 128, 64],
            beta: 4.0,
            dropout: 0.1,
            sequence_length: 60,
            n_regimes: 3,
            seed: 42,
        }
    }
}

impl VAEConfig {
    /// Create a new configuration with custom parameters
    pub fn new(
        input_dim: usize,
        latent_dim: usize,
        hidden_dims: Vec<usize>,
    ) -> Self {
        Self {
            input_dim,
            latent_dim,
            hidden_dims,
            ..Default::default()
        }
    }

    /// Set beta coefficient
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Set sequence length
    pub fn with_sequence_length(mut self, seq_len: usize) -> Self {
        self.sequence_length = seq_len;
        self
    }

    /// Set number of regimes
    pub fn with_n_regimes(mut self, n_regimes: usize) -> Self {
        self.n_regimes = n_regimes;
        self
    }

    /// Get encoder output dimension
    pub fn encoder_output_dim(&self) -> usize {
        *self.hidden_dims.last().unwrap_or(&self.input_dim) * 2
    }

    /// Get decoder input dimension
    pub fn decoder_input_dim(&self) -> usize {
        self.latent_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = VAEConfig::default();
        assert_eq!(config.latent_dim, 16);
        assert_eq!(config.beta, 4.0);
    }

    #[test]
    fn test_builder_pattern() {
        let config = VAEConfig::new(20, 32, vec![128, 64])
            .with_beta(2.0)
            .with_sequence_length(100);

        assert_eq!(config.input_dim, 20);
        assert_eq!(config.latent_dim, 32);
        assert_eq!(config.beta, 2.0);
        assert_eq!(config.sequence_length, 100);
    }
}
