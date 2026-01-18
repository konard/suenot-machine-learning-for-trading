//! Model module for Cross-Attention Multi-Asset Trading
//!
//! This module provides the core model components:
//! - Attention mechanisms (self-attention and cross-attention)
//! - Token embeddings
//! - The main CrossAttentionMultiAsset model

mod attention;
mod embedding;
mod cross_attention;

pub use attention::{MultiHeadAttention, CrossAssetAttention, TemporalAttention};
pub use embedding::{TokenEmbedding, PositionalEncoding, AssetEmbedding};
pub use cross_attention::{CrossAttentionMultiAsset, CrossAttentionEncoderLayer};

use serde::{Deserialize, Serialize};

/// Output type for the model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputType {
    /// Regression: predict continuous returns
    Regression,
    /// Classification: predict direction (up/down)
    Classification,
    /// Portfolio: output portfolio weights
    Portfolio,
}

impl Default for OutputType {
    fn default() -> Self {
        OutputType::Portfolio
    }
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Number of assets
    pub n_assets: usize,
    /// Number of input features per asset
    pub n_features: usize,
    /// Sequence length (lookback window)
    pub seq_len: usize,
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of encoder layers
    pub n_layers: usize,
    /// Feed-forward dimension
    pub d_ff: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Output type
    pub output_type: OutputType,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_assets: 5,
            n_features: 6,
            seq_len: 168,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            d_ff: 256,
            dropout: 0.1,
            output_type: OutputType::Portfolio,
        }
    }
}

impl ModelConfig {
    /// Create a new configuration with custom parameters
    pub fn new(
        n_assets: usize,
        n_features: usize,
        seq_len: usize,
        d_model: usize,
        n_heads: usize,
    ) -> Self {
        Self {
            n_assets,
            n_features,
            seq_len,
            d_model,
            n_heads,
            n_layers: 2,
            d_ff: d_model * 4,
            dropout: 0.1,
            output_type: OutputType::Portfolio,
        }
    }

    /// Set the output type
    pub fn with_output_type(mut self, output_type: OutputType) -> Self {
        self.output_type = output_type;
        self
    }

    /// Set the number of layers
    pub fn with_n_layers(mut self, n_layers: usize) -> Self {
        self.n_layers = n_layers;
        self
    }

    /// Set the dropout rate
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.d_model % self.n_heads != 0 {
            return Err(format!(
                "d_model ({}) must be divisible by n_heads ({})",
                self.d_model, self.n_heads
            ));
        }
        if self.n_assets == 0 {
            return Err("n_assets must be greater than 0".to_string());
        }
        if self.n_features == 0 {
            return Err("n_features must be greater than 0".to_string());
        }
        if self.seq_len == 0 {
            return Err("seq_len must be greater than 0".to_string());
        }
        if self.dropout < 0.0 || self.dropout > 1.0 {
            return Err(format!("dropout ({}) must be between 0 and 1", self.dropout));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ModelConfig::default();
        assert_eq!(config.n_assets, 5);
        assert_eq!(config.n_features, 6);
        assert_eq!(config.d_model, 64);
        assert_eq!(config.n_heads, 4);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        let mut config = ModelConfig::default();

        // Invalid: d_model not divisible by n_heads
        config.d_model = 63;
        assert!(config.validate().is_err());

        // Invalid: zero assets
        config = ModelConfig::default();
        config.n_assets = 0;
        assert!(config.validate().is_err());

        // Invalid: dropout out of range
        config = ModelConfig::default();
        config.dropout = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_output_type_default() {
        assert_eq!(OutputType::default(), OutputType::Portfolio);
    }
}
