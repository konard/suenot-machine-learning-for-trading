//! Higher Order Transformer model
//!
//! Full transformer architecture with higher-order attention layers

use ndarray::Array2;
use rand::Rng;
use rand_distr::StandardNormal;

use crate::attention::HigherOrderAttention;

/// Layer normalization
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Normalized dimension
    pub dim: usize,
    /// Scale parameters (gamma)
    pub gamma: ndarray::Array1<f64>,
    /// Shift parameters (beta)
    pub beta: ndarray::Array1<f64>,
    /// Epsilon for numerical stability
    pub eps: f64,
}

impl LayerNorm {
    /// Create a new layer norm
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            gamma: ndarray::Array1::ones(dim),
            beta: ndarray::Array1::zeros(dim),
            eps: 1e-6,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut result = x.clone();

        for i in 0..x.nrows() {
            let row = x.row(i);
            let mean = row.mean().unwrap_or(0.0);
            let var: f64 = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / self.dim as f64;
            let std = (var + self.eps).sqrt();

            for j in 0..self.dim {
                result[[i, j]] = ((x[[i, j]] - mean) / std) * self.gamma[j] + self.beta[j];
            }
        }

        result
    }
}

/// Feed-forward network
#[derive(Debug, Clone)]
pub struct FeedForward {
    /// Input/output dimension
    pub d_model: usize,
    /// Hidden dimension
    pub d_ff: usize,
    /// First linear layer
    pub w1: Array2<f64>,
    /// Second linear layer
    pub w2: Array2<f64>,
    /// First bias
    pub b1: ndarray::Array1<f64>,
    /// Second bias
    pub b2: ndarray::Array1<f64>,
}

impl FeedForward {
    /// Create a new feed-forward network
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let mut rng = rand::thread_rng();

        let init_weight = |rows: usize, cols: usize| -> Array2<f64> {
            let std = (2.0 / (rows + cols) as f64).sqrt();
            Array2::from_shape_fn((rows, cols), |_| rng.sample::<f64, _>(StandardNormal) * std)
        };

        Self {
            d_model,
            d_ff,
            w1: init_weight(d_model, d_ff),
            w2: init_weight(d_ff, d_model),
            b1: ndarray::Array1::zeros(d_ff),
            b2: ndarray::Array1::zeros(d_model),
        }
    }

    /// Forward pass with GELU activation
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        // First linear layer
        let h = x.dot(&self.w1) + &self.b1;

        // GELU activation: x * Phi(x) where Phi is the CDF of standard normal
        let h = h.mapv(|v| v * 0.5 * (1.0 + (v / std::f64::consts::SQRT_2).tanh()));

        // Second linear layer
        h.dot(&self.w2) + &self.b2
    }
}

/// A single HOT block (attention + feed-forward)
#[derive(Debug, Clone)]
pub struct HOTBlock {
    /// Higher-order attention
    pub attention: HigherOrderAttention,
    /// Feed-forward network
    pub ff: FeedForward,
    /// Layer norm before attention
    pub norm1: LayerNorm,
    /// Layer norm before feed-forward
    pub norm2: LayerNorm,
    /// Dropout rate (stored but not applied in this simple version)
    pub dropout: f64,
}

impl HOTBlock {
    /// Create a new HOT block
    pub fn new(d_model: usize, n_heads: usize, rank: usize, d_ff: usize, dropout: f64) -> Self {
        Self {
            attention: HigherOrderAttention::new(d_model, n_heads, rank),
            ff: FeedForward::new(d_model, d_ff),
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
            dropout,
        }
    }

    /// Forward pass with residual connections
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        // Pre-norm architecture
        // Attention block
        let normed = self.norm1.forward(x);
        let attn_out = self.attention.forward(&normed);
        let x = x + &attn_out; // Residual connection

        // Feed-forward block
        let normed = self.norm2.forward(&x);
        let ff_out = self.ff.forward(&normed);
        &x + &ff_out // Residual connection
    }
}

/// Full Higher Order Transformer model
#[derive(Debug, Clone)]
pub struct HOTModel {
    /// Input projection
    pub input_proj: Array2<f64>,
    /// Model dimension
    pub d_model: usize,
    /// HOT blocks
    pub blocks: Vec<HOTBlock>,
    /// Final layer norm
    pub final_norm: LayerNorm,
    /// Output dimension
    pub output_dim: usize,
}

impl HOTModel {
    /// Create a new HOT model
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Input feature dimension
    /// * `d_model` - Model hidden dimension
    /// * `n_heads` - Number of attention heads
    /// * `n_layers` - Number of HOT blocks
    /// * `rank` - CP decomposition rank
    /// * `output_dim` - Output dimension (e.g., 3 for up/down/neutral)
    pub fn new(
        input_dim: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        rank: usize,
        output_dim: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();

        let d_ff = d_model * 4;
        let dropout = 0.1;

        // Input projection
        let std = (2.0 / (input_dim + d_model) as f64).sqrt();
        let input_proj = Array2::from_shape_fn((input_dim, d_model), |_| {
            rng.sample::<f64, _>(StandardNormal) * std
        });

        // HOT blocks
        let blocks: Vec<HOTBlock> = (0..n_layers)
            .map(|_| HOTBlock::new(d_model, n_heads, rank, d_ff, dropout))
            .collect();

        Self {
            input_proj,
            d_model,
            blocks,
            final_norm: LayerNorm::new(d_model),
            output_dim,
        }
    }

    /// Create with default configuration for crypto trading
    pub fn for_crypto(input_dim: usize) -> Self {
        Self::new(
            input_dim,
            128,  // d_model
            4,    // n_heads
            3,    // n_layers
            16,   // rank
            3,    // output_dim (up/down/neutral)
        )
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape (seq_len, input_dim)
    ///
    /// # Returns
    ///
    /// Output tensor of shape (seq_len, d_model)
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        // Project input to model dimension
        let mut h = x.dot(&self.input_proj);

        // Apply positional encoding (simple learnable)
        let seq_len = h.nrows();
        for i in 0..seq_len {
            for j in 0..self.d_model {
                // Simple sinusoidal position encoding
                let pos = i as f64;
                let dim = j as f64;
                if j % 2 == 0 {
                    h[[i, j]] += (pos / 10000_f64.powf(dim / self.d_model as f64)).sin();
                } else {
                    h[[i, j]] += (pos / 10000_f64.powf((dim - 1.0) / self.d_model as f64)).cos();
                }
            }
        }

        // Apply HOT blocks
        for block in &self.blocks {
            h = block.forward(&h);
        }

        // Final normalization
        self.final_norm.forward(&h)
    }

    /// Get the representation for the last time step (for classification)
    pub fn get_last_representation(&self, x: &Array2<f64>) -> ndarray::Array1<f64> {
        let output = self.forward(x);
        output.row(output.nrows() - 1).to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(64);
        let x = Array2::from_shape_fn((10, 64), |_| rand::random::<f64>());

        let output = ln.forward(&x);
        assert_eq!(output.shape(), &[10, 64]);
    }

    #[test]
    fn test_feed_forward() {
        let ff = FeedForward::new(64, 256);
        let x = Array2::from_shape_fn((10, 64), |_| rand::random::<f64>());

        let output = ff.forward(&x);
        assert_eq!(output.shape(), &[10, 64]);
    }

    #[test]
    fn test_hot_model() {
        let model = HOTModel::for_crypto(7); // 7 features
        let x = Array2::from_shape_fn((60, 7), |_| rand::random::<f64>());

        let output = model.forward(&x);
        assert_eq!(output.nrows(), 60);
        assert_eq!(output.ncols(), 128); // d_model
    }
}
