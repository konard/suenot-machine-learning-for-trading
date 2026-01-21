//! Shared encoder architectures for task-agnostic learning
//!
//! This module provides various encoder architectures that create
//! universal market representations:
//!
//! - Transformer encoder with self-attention
//! - CNN encoder for local pattern extraction
//! - Mixture of Experts (MoE) for specialized processing

mod transformer;
mod cnn;
mod moe;

pub use transformer::{TransformerEncoder, TransformerConfig};
pub use cnn::{CNNEncoder, CNNConfig};
pub use moe::{MoEEncoder, MoEConfig};

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Encoder type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncoderType {
    /// Transformer with self-attention
    Transformer,
    /// Convolutional neural network
    CNN,
    /// Mixture of Experts
    MoE,
}

impl Default for EncoderType {
    fn default() -> Self {
        Self::Transformer
    }
}

/// Configuration for shared encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    /// Type of encoder to use
    pub encoder_type: EncoderType,
    /// Input feature dimension
    pub input_dim: usize,
    /// Output embedding dimension
    pub embedding_dim: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Dropout rate
    pub dropout: f64,
    /// Whether to use layer normalization
    pub use_layer_norm: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            encoder_type: EncoderType::Transformer,
            input_dim: 20,
            embedding_dim: 64,
            hidden_dims: vec![128, 64],
            dropout: 0.1,
            use_layer_norm: true,
        }
    }
}

impl EncoderConfig {
    /// Set encoder type
    pub fn with_encoder_type(mut self, encoder_type: EncoderType) -> Self {
        self.encoder_type = encoder_type;
        self
    }

    /// Set input dimension
    pub fn with_input_dim(mut self, input_dim: usize) -> Self {
        self.input_dim = input_dim;
        self
    }

    /// Set embedding dimension
    pub fn with_embedding_dim(mut self, embedding_dim: usize) -> Self {
        self.embedding_dim = embedding_dim;
        self
    }

    /// Set hidden dimensions
    pub fn with_hidden_dims(mut self, hidden_dims: Vec<usize>) -> Self {
        self.hidden_dims = hidden_dims;
        self
    }

    /// Set dropout rate
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }
}

/// Trait for shared encoders
pub trait SharedEncoder: Send + Sync {
    /// Encode a single input to embedding
    fn encode(&self, input: &Array1<f64>) -> Array1<f64>;

    /// Encode a batch of inputs
    fn encode_batch(&self, inputs: &Array2<f64>) -> Array2<f64>;

    /// Get the embedding dimension
    fn embedding_dim(&self) -> usize;

    /// Get encoder parameters for gradient updates
    fn parameters(&self) -> Vec<Array2<f64>>;

    /// Update encoder parameters
    fn update_parameters(&mut self, gradients: &[Array2<f64>], learning_rate: f64);
}

/// Factory function to create encoder based on config
pub fn create_encoder(config: &EncoderConfig) -> Box<dyn SharedEncoder> {
    match config.encoder_type {
        EncoderType::Transformer => {
            let transformer_config = TransformerConfig {
                input_dim: config.input_dim,
                embedding_dim: config.embedding_dim,
                num_heads: 4,
                num_layers: 2,
                ff_dim: config.hidden_dims.first().copied().unwrap_or(128),
                dropout: config.dropout,
                use_layer_norm: config.use_layer_norm,
            };
            Box::new(TransformerEncoder::new(transformer_config))
        }
        EncoderType::CNN => {
            let cnn_config = CNNConfig {
                input_dim: config.input_dim,
                embedding_dim: config.embedding_dim,
                kernel_sizes: vec![3, 5, 7],
                num_filters: 32,
                dropout: config.dropout,
            };
            Box::new(CNNEncoder::new(cnn_config))
        }
        EncoderType::MoE => {
            let moe_config = MoEConfig {
                input_dim: config.input_dim,
                embedding_dim: config.embedding_dim,
                num_experts: 4,
                expert_dim: config.hidden_dims.first().copied().unwrap_or(64),
                top_k: 2,
                dropout: config.dropout,
            };
            Box::new(MoEEncoder::new(moe_config))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = EncoderConfig::default();
        assert_eq!(config.encoder_type, EncoderType::Transformer);
        assert_eq!(config.embedding_dim, 64);
    }

    #[test]
    fn test_config_builder() {
        let config = EncoderConfig::default()
            .with_encoder_type(EncoderType::CNN)
            .with_embedding_dim(128);
        assert_eq!(config.encoder_type, EncoderType::CNN);
        assert_eq!(config.embedding_dim, 128);
    }

    #[test]
    fn test_create_encoder() {
        let config = EncoderConfig::default().with_input_dim(10);
        let encoder = create_encoder(&config);
        assert_eq!(encoder.embedding_dim(), 64);
    }
}
