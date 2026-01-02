//! Graph Transformer configuration
//!
//! Defines hyperparameters and configuration for Graph Transformer models.

use serde::{Deserialize, Serialize};

/// Configuration for Graph Transformer model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTransformerConfig {
    /// Input feature dimension
    pub input_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Edge feature dimension (None = no edge features)
    pub edge_dim: Option<usize>,
    /// Whether to use edge features in attention
    pub use_edge_features: bool,
    /// Type of positional encoding
    pub positional_encoding: PositionalEncodingType,
    /// Number of positional encoding dimensions
    pub num_pe_dims: usize,
    /// Activation function
    pub activation: ActivationType,
    /// Whether to use layer normalization
    pub use_layer_norm: bool,
    /// Whether to use residual connections
    pub use_residual: bool,
    /// Feed-forward expansion ratio
    pub ff_expansion: usize,
}

impl Default for GraphTransformerConfig {
    fn default() -> Self {
        Self {
            input_dim: 32,
            hidden_dim: 64,
            output_dim: 3,  // up/neutral/down
            num_layers: 4,
            num_heads: 4,
            dropout: 0.1,
            edge_dim: Some(8),
            use_edge_features: true,
            positional_encoding: PositionalEncodingType::Laplacian,
            num_pe_dims: 8,
            activation: ActivationType::GELU,
            use_layer_norm: true,
            use_residual: true,
            ff_expansion: 4,
        }
    }
}

impl GraphTransformerConfig {
    /// Create a minimal config for testing
    pub fn minimal() -> Self {
        Self {
            input_dim: 8,
            hidden_dim: 16,
            output_dim: 3,
            num_layers: 2,
            num_heads: 2,
            dropout: 0.0,
            edge_dim: None,
            use_edge_features: false,
            positional_encoding: PositionalEncodingType::None,
            num_pe_dims: 0,
            activation: ActivationType::ReLU,
            use_layer_norm: false,
            use_residual: true,
            ff_expansion: 2,
        }
    }

    /// Create a standard config for production
    pub fn standard() -> Self {
        Self::default()
    }

    /// Create a large config for better performance
    pub fn large() -> Self {
        Self {
            input_dim: 64,
            hidden_dim: 256,
            output_dim: 3,
            num_layers: 6,
            num_heads: 8,
            dropout: 0.1,
            edge_dim: Some(32),
            use_edge_features: true,
            positional_encoding: PositionalEncodingType::LaplacianRW,
            num_pe_dims: 16,
            activation: ActivationType::GELU,
            use_layer_norm: true,
            use_residual: true,
            ff_expansion: 4,
        }
    }

    /// Get the total feature dimension after adding positional encoding
    pub fn total_feature_dim(&self) -> usize {
        self.input_dim + self.num_pe_dims
    }

    /// Get feed-forward hidden dimension
    pub fn ff_hidden_dim(&self) -> usize {
        self.hidden_dim * self.ff_expansion
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_dim % self.num_heads != 0 {
            return Err(format!(
                "hidden_dim ({}) must be divisible by num_heads ({})",
                self.hidden_dim, self.num_heads
            ));
        }

        if self.num_layers == 0 {
            return Err("num_layers must be at least 1".to_string());
        }

        if self.dropout < 0.0 || self.dropout >= 1.0 {
            return Err("dropout must be in [0, 1)".to_string());
        }

        Ok(())
    }

    /// Builder method: set hidden dimension
    pub fn with_hidden_dim(mut self, dim: usize) -> Self {
        self.hidden_dim = dim;
        self
    }

    /// Builder method: set number of layers
    pub fn with_num_layers(mut self, n: usize) -> Self {
        self.num_layers = n;
        self
    }

    /// Builder method: set number of heads
    pub fn with_num_heads(mut self, n: usize) -> Self {
        self.num_heads = n;
        self
    }

    /// Builder method: set dropout
    pub fn with_dropout(mut self, d: f64) -> Self {
        self.dropout = d;
        self
    }

    /// Builder method: set positional encoding type
    pub fn with_positional_encoding(mut self, pe: PositionalEncodingType) -> Self {
        self.positional_encoding = pe;
        self
    }
}

/// Type of positional encoding to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionalEncodingType {
    /// No positional encoding
    None,
    /// Laplacian eigenvector encoding
    Laplacian,
    /// Random walk positional encoding
    RandomWalk,
    /// Combined Laplacian and Random Walk
    LaplacianRW,
    /// Degree-based encoding
    Degree,
    /// Custom learnable encoding
    Learnable,
}

impl PositionalEncodingType {
    /// Check if encoding requires eigenvalue computation
    pub fn requires_eigenvectors(&self) -> bool {
        matches!(self, Self::Laplacian | Self::LaplacianRW)
    }

    /// Check if encoding requires random walk computation
    pub fn requires_random_walk(&self) -> bool {
        matches!(self, Self::RandomWalk | Self::LaplacianRW)
    }
}

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationType {
    /// Rectified Linear Unit
    ReLU,
    /// Gaussian Error Linear Unit
    GELU,
    /// Exponential Linear Unit
    ELU,
    /// Sigmoid Linear Unit (Swish)
    SiLU,
    /// Hyperbolic tangent
    Tanh,
}

impl ActivationType {
    /// Apply activation function
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Self::ReLU => x.max(0.0),
            Self::GELU => {
                // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                let sqrt_2_over_pi = 0.7978845608;
                0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
            }
            Self::ELU => {
                if x > 0.0 {
                    x
                } else {
                    x.exp() - 1.0
                }
            }
            Self::SiLU => x * (1.0 / (1.0 + (-x).exp())),  // x * sigmoid(x)
            Self::Tanh => x.tanh(),
        }
    }

    /// Apply activation function to array (in-place)
    pub fn apply_inplace(&self, data: &mut [f64]) {
        for x in data.iter_mut() {
            *x = self.apply(*x);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GraphTransformerConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_config() {
        let config = GraphTransformerConfig {
            hidden_dim: 65,  // Not divisible by 4
            num_heads: 4,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_activations() {
        let relu = ActivationType::ReLU;
        assert_eq!(relu.apply(-1.0), 0.0);
        assert_eq!(relu.apply(1.0), 1.0);

        let gelu = ActivationType::GELU;
        assert!(gelu.apply(1.0) > 0.0);
        assert!(gelu.apply(-1.0) < 0.0);
    }
}
