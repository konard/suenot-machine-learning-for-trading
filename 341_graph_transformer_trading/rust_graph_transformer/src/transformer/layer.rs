//! Graph Transformer layer
//!
//! A single layer of the Graph Transformer combining attention and feed-forward networks.

use super::attention::GraphAttention;
use super::config::{ActivationType, GraphTransformerConfig};
use ndarray::{Array1, Array2};
use rand::Rng;

/// Single Graph Transformer layer
#[derive(Debug, Clone)]
pub struct GraphTransformerLayer {
    /// Graph attention sublayer
    pub attention: GraphAttention,
    /// Feed-forward network weights
    pub ff_w1: Array2<f64>,
    pub ff_w2: Array2<f64>,
    /// Layer normalization parameters
    pub ln1_gamma: Array1<f64>,
    pub ln1_beta: Array1<f64>,
    pub ln2_gamma: Array1<f64>,
    pub ln2_beta: Array1<f64>,
    /// Configuration
    pub hidden_dim: usize,
    pub ff_dim: usize,
    pub activation: ActivationType,
    pub use_layer_norm: bool,
    pub use_residual: bool,
    pub dropout: f64,
}

impl GraphTransformerLayer {
    /// Create a new Graph Transformer layer
    pub fn new(config: &GraphTransformerConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        let ff_dim = config.ff_hidden_dim();
        let mut rng = rand::thread_rng();

        // Initialize attention
        let attention = GraphAttention::from_config(config);

        // Initialize feed-forward weights
        let ff_xavier = (2.0 / (hidden_dim + ff_dim) as f64).sqrt();
        let ff_w1 = Self::random_matrix(hidden_dim, ff_dim, ff_xavier, &mut rng);
        let ff_w2 = Self::random_matrix(ff_dim, hidden_dim, ff_xavier, &mut rng);

        // Initialize layer norm parameters
        let ln1_gamma = Array1::ones(hidden_dim);
        let ln1_beta = Array1::zeros(hidden_dim);
        let ln2_gamma = Array1::ones(hidden_dim);
        let ln2_beta = Array1::zeros(hidden_dim);

        Self {
            attention,
            ff_w1,
            ff_w2,
            ln1_gamma,
            ln1_beta,
            ln2_gamma,
            ln2_beta,
            hidden_dim,
            ff_dim,
            activation: config.activation,
            use_layer_norm: config.use_layer_norm,
            use_residual: config.use_residual,
            dropout: config.dropout,
        }
    }

    fn random_matrix(rows: usize, cols: usize, std: f64, rng: &mut impl Rng) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |_| rng.gen::<f64>() * std * 2.0 - std)
    }

    /// Layer normalization
    fn layer_norm(&self, x: &Array2<f64>, gamma: &Array1<f64>, beta: &Array1<f64>) -> Array2<f64> {
        let eps = 1e-6;
        let mut result = Array2::zeros(x.raw_dim());

        for i in 0..x.nrows() {
            let row = x.row(i);
            let mean = row.mean().unwrap_or(0.0);
            let var = row.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(1.0);
            let std = (var + eps).sqrt();

            for j in 0..x.ncols() {
                result[[i, j]] = gamma[j] * (x[[i, j]] - mean) / std + beta[j];
            }
        }

        result
    }

    /// Feed-forward network
    fn feed_forward(&self, x: &Array2<f64>) -> Array2<f64> {
        // First linear layer
        let mut hidden = x.dot(&self.ff_w1);

        // Apply activation
        hidden.mapv_inplace(|v| self.activation.apply(v));

        // Second linear layer
        hidden.dot(&self.ff_w2)
    }

    /// Forward pass through the layer
    pub fn forward(
        &self,
        x: &Array2<f64>,
        edge_index: &Array2<usize>,
        edge_attr: Option<&Array2<f64>>,
    ) -> Array2<f64> {
        // Attention sublayer
        let attn_out = self.attention.forward(x, edge_index, edge_attr);

        // Residual connection + layer norm
        let x1 = if self.use_residual {
            x + &attn_out
        } else {
            attn_out
        };

        let x1_norm = if self.use_layer_norm {
            self.layer_norm(&x1, &self.ln1_gamma, &self.ln1_beta)
        } else {
            x1
        };

        // Feed-forward sublayer
        let ff_out = self.feed_forward(&x1_norm);

        // Residual connection + layer norm
        let x2 = if self.use_residual {
            &x1_norm + &ff_out
        } else {
            ff_out
        };

        if self.use_layer_norm {
            self.layer_norm(&x2, &self.ln2_gamma, &self.ln2_beta)
        } else {
            x2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_forward() {
        let config = GraphTransformerConfig::minimal();
        let layer = GraphTransformerLayer::new(&config);

        let x = Array2::from_shape_fn((3, 16), |_| rand::random::<f64>());
        let edge_index = Array2::from_shape_vec((2, 2), vec![0, 1, 1, 2]).unwrap();

        let output = layer.forward(&x, &edge_index, None);
        assert_eq!(output.shape(), &[3, 16]);
    }
}
