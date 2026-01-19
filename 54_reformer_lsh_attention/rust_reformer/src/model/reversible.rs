//! Reversible layers for memory-efficient training
//!
//! Implements reversible residual connections that allow recomputation
//! of activations during backpropagation, reducing memory usage.

use ndarray::{Array2, Array3};
use std::f64::consts::PI;

use super::config::ReformerConfig;
use super::lsh_attention::LSHAttention;

/// Chunked feed-forward network
///
/// Processes input in chunks to reduce memory usage for long sequences.
#[derive(Debug, Clone)]
pub struct ChunkedFeedForward {
    /// First linear layer [d_model, d_ff]
    w1: Array2<f64>,
    /// Second linear layer [d_ff, d_model]
    w2: Array2<f64>,
    /// Chunk size
    chunk_size: usize,
    /// Dropout probability (not applied in inference)
    dropout: f64,
}

impl ChunkedFeedForward {
    /// Create a new chunked feed-forward layer
    pub fn new(config: &ReformerConfig) -> Self {
        let d_model = config.d_model;
        let d_ff = config.d_ff;

        // Xavier initialization
        let scale1 = (2.0 / (d_model + d_ff) as f64).sqrt();
        let scale2 = (2.0 / (d_ff + d_model) as f64).sqrt();

        let w1 = Array2::from_shape_fn((d_model, d_ff), |_| rand_normal() * scale1);
        let w2 = Array2::from_shape_fn((d_ff, d_model), |_| rand_normal() * scale2);

        Self {
            w1,
            w2,
            chunk_size: config.chunk_size,
            dropout: config.dropout,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let (seq_len, d_model) = x.dim();
        let d_ff = self.w1.ncols();

        let mut output = Array2::zeros((seq_len, d_model));

        // Process in chunks
        let n_chunks = (seq_len + self.chunk_size - 1) / self.chunk_size;

        for chunk_idx in 0..n_chunks {
            let start = chunk_idx * self.chunk_size;
            let end = (start + self.chunk_size).min(seq_len);
            let chunk_len = end - start;

            // First linear + GELU activation
            for t in start..end {
                for d in 0..d_ff {
                    let mut sum = 0.0;
                    for d_in in 0..d_model {
                        sum += x[[t, d_in]] * self.w1[[d_in, d]];
                    }
                    // GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    let gelu = gelu_activation(sum);

                    // Second linear
                    for d_out in 0..d_model {
                        output[[t, d_out]] += gelu * self.w2[[d, d_out]];
                    }
                }
            }
        }

        output
    }

    /// Forward pass for 3D tensor
    pub fn forward_3d(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, d_model) = x.dim();

        let mut output = Array3::zeros((batch_size, seq_len, d_model));

        for b in 0..batch_size {
            let x_2d = x.index_axis(ndarray::Axis(0), b).to_owned();
            let out_2d = self.forward(&x_2d);

            for t in 0..seq_len {
                for d in 0..d_model {
                    output[[b, t, d]] = out_2d[[t, d]];
                }
            }
        }

        output
    }
}

/// Reversible block combining attention and feed-forward
///
/// Uses the reversible residual formulation:
/// Y1 = X1 + Attention(X2)
/// Y2 = X2 + FFN(Y1)
///
/// Which can be reversed:
/// X2 = Y2 - FFN(Y1)
/// X1 = Y1 - Attention(X2)
#[derive(Debug)]
pub struct ReversibleBlock {
    /// LSH Attention layer
    attention: LSHAttention,
    /// Feed-forward layer
    ffn: ChunkedFeedForward,
    /// Layer normalization parameters (gamma) for attention
    ln1_gamma: Array2<f64>,
    /// Layer normalization parameters (beta) for attention
    ln1_beta: Array2<f64>,
    /// Layer normalization parameters (gamma) for FFN
    ln2_gamma: Array2<f64>,
    /// Layer normalization parameters (beta) for FFN
    ln2_beta: Array2<f64>,
    /// Model dimension
    d_model: usize,
}

impl ReversibleBlock {
    /// Create a new reversible block
    pub fn new(config: &ReformerConfig) -> Self {
        let d_model = config.d_model;

        Self {
            attention: LSHAttention::new(config),
            ffn: ChunkedFeedForward::new(config),
            ln1_gamma: Array2::ones((1, d_model)),
            ln1_beta: Array2::zeros((1, d_model)),
            ln2_gamma: Array2::ones((1, d_model)),
            ln2_beta: Array2::zeros((1, d_model)),
            d_model,
        }
    }

    /// Forward pass for reversible block
    ///
    /// # Arguments
    /// * `x1` - First input stream [batch, seq_len, d_model]
    /// * `x2` - Second input stream [batch, seq_len, d_model]
    ///
    /// # Returns
    /// * `y1` - First output stream
    /// * `y2` - Second output stream
    pub fn forward(&self, x1: &Array3<f64>, x2: &Array3<f64>) -> (Array3<f64>, Array3<f64>) {
        // Y1 = X1 + Attention(LayerNorm(X2))
        let ln_x2 = self.layer_norm_3d(x2, &self.ln1_gamma, &self.ln1_beta);
        let (attn_out, _) = self.attention.forward(&ln_x2);
        let y1 = add_3d(x1, &attn_out);

        // Y2 = X2 + FFN(LayerNorm(Y1))
        let ln_y1 = self.layer_norm_3d(&y1, &self.ln2_gamma, &self.ln2_beta);
        let ffn_out = self.ffn.forward_3d(&ln_y1);
        let y2 = add_3d(x2, &ffn_out);

        (y1, y2)
    }

    /// Reverse pass (for gradient computation)
    ///
    /// Given outputs, reconstruct inputs without storing activations
    pub fn reverse(&self, y1: &Array3<f64>, y2: &Array3<f64>) -> (Array3<f64>, Array3<f64>) {
        // X2 = Y2 - FFN(LayerNorm(Y1))
        let ln_y1 = self.layer_norm_3d(y1, &self.ln2_gamma, &self.ln2_beta);
        let ffn_out = self.ffn.forward_3d(&ln_y1);
        let x2 = sub_3d(y2, &ffn_out);

        // X1 = Y1 - Attention(LayerNorm(X2))
        let ln_x2 = self.layer_norm_3d(&x2, &self.ln1_gamma, &self.ln1_beta);
        let (attn_out, _) = self.attention.forward(&ln_x2);
        let x1 = sub_3d(y1, &attn_out);

        (x1, x2)
    }

    /// Forward pass for 2D input
    pub fn forward_2d(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        // Layer norm + attention
        let ln_x2 = self.layer_norm_2d(x2, &self.ln1_gamma, &self.ln1_beta);
        let (attn_out, _) = self.attention.forward_2d(&ln_x2);
        let y1 = add_2d(x1, &attn_out);

        // Layer norm + FFN
        let ln_y1 = self.layer_norm_2d(&y1, &self.ln2_gamma, &self.ln2_beta);
        let ffn_out = self.ffn.forward(&ln_y1);
        let y2 = add_2d(x2, &ffn_out);

        (y1, y2)
    }

    /// Layer normalization for 3D tensor
    fn layer_norm_3d(
        &self,
        x: &Array3<f64>,
        gamma: &Array2<f64>,
        beta: &Array2<f64>,
    ) -> Array3<f64> {
        let (batch_size, seq_len, d_model) = x.dim();
        let mut output = Array3::zeros((batch_size, seq_len, d_model));

        for b in 0..batch_size {
            for t in 0..seq_len {
                // Compute mean and variance
                let mut sum = 0.0;
                for d in 0..d_model {
                    sum += x[[b, t, d]];
                }
                let mean = sum / d_model as f64;

                let mut var_sum = 0.0;
                for d in 0..d_model {
                    let diff = x[[b, t, d]] - mean;
                    var_sum += diff * diff;
                }
                let std = (var_sum / d_model as f64 + 1e-6).sqrt();

                // Normalize and scale
                for d in 0..d_model {
                    let normalized = (x[[b, t, d]] - mean) / std;
                    output[[b, t, d]] = gamma[[0, d]] * normalized + beta[[0, d]];
                }
            }
        }

        output
    }

    /// Layer normalization for 2D tensor
    fn layer_norm_2d(
        &self,
        x: &Array2<f64>,
        gamma: &Array2<f64>,
        beta: &Array2<f64>,
    ) -> Array2<f64> {
        let (seq_len, d_model) = x.dim();
        let mut output = Array2::zeros((seq_len, d_model));

        for t in 0..seq_len {
            let mut sum = 0.0;
            for d in 0..d_model {
                sum += x[[t, d]];
            }
            let mean = sum / d_model as f64;

            let mut var_sum = 0.0;
            for d in 0..d_model {
                let diff = x[[t, d]] - mean;
                var_sum += diff * diff;
            }
            let std = (var_sum / d_model as f64 + 1e-6).sqrt();

            for d in 0..d_model {
                let normalized = (x[[t, d]] - mean) / std;
                output[[t, d]] = gamma[[0, d]] * normalized + beta[[0, d]];
            }
        }

        output
    }
}

/// GELU activation function
fn gelu_activation(x: f64) -> f64 {
    let sqrt_2_pi = (2.0 / PI).sqrt();
    x * 0.5 * (1.0 + (sqrt_2_pi * (x + 0.044715 * x.powi(3))).tanh())
}

/// Add two 3D arrays element-wise
fn add_3d(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
    a + b
}

/// Subtract two 3D arrays element-wise
fn sub_3d(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
    a - b
}

/// Add two 2D arrays element-wise
fn add_2d(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    a + b
}

/// Generate random number from standard normal distribution
fn rand_normal() -> f64 {
    let u1: f64 = rand::random::<f64>().max(1e-10);
    let u2: f64 = rand::random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> ReformerConfig {
        ReformerConfig {
            d_model: 32,
            d_ff: 64,
            n_heads: 4,
            n_hash_rounds: 2,
            n_buckets: 8,
            chunk_size: 8,
            seq_len: 32,
            ..Default::default()
        }
    }

    #[test]
    fn test_chunked_ffn() {
        let config = create_test_config();
        let ffn = ChunkedFeedForward::new(&config);

        let x = Array2::from_shape_fn((16, 32), |_| rand_normal());
        let output = ffn.forward(&x);

        assert_eq!(output.dim(), (16, 32));
    }

    #[test]
    fn test_reversible_block_forward() {
        let config = create_test_config();
        let block = ReversibleBlock::new(&config);

        let x1 = Array3::from_shape_fn((1, 16, 32), |_| rand_normal() * 0.1);
        let x2 = Array3::from_shape_fn((1, 16, 32), |_| rand_normal() * 0.1);

        let (y1, y2) = block.forward(&x1, &x2);

        assert_eq!(y1.dim(), (1, 16, 32));
        assert_eq!(y2.dim(), (1, 16, 32));
    }

    #[test]
    fn test_reversible_block_reversibility() {
        let config = create_test_config();
        let block = ReversibleBlock::new(&config);

        let x1_orig = Array3::from_shape_fn((1, 16, 32), |_| rand_normal() * 0.1);
        let x2_orig = Array3::from_shape_fn((1, 16, 32), |_| rand_normal() * 0.1);

        // Forward
        let (y1, y2) = block.forward(&x1_orig, &x2_orig);

        // Reverse
        let (x1_recovered, x2_recovered) = block.reverse(&y1, &y2);

        // Check reconstruction (should be close, not exact due to floating point)
        let max_diff_x1 = (&x1_orig - &x1_recovered)
            .iter()
            .map(|x| x.abs())
            .fold(0.0, f64::max);

        let max_diff_x2 = (&x2_orig - &x2_recovered)
            .iter()
            .map(|x| x.abs())
            .fold(0.0, f64::max);

        assert!(
            max_diff_x1 < 1e-10,
            "X1 reconstruction error too large: {}",
            max_diff_x1
        );
        assert!(
            max_diff_x2 < 1e-10,
            "X2 reconstruction error too large: {}",
            max_diff_x2
        );
    }

    #[test]
    fn test_gelu_activation() {
        // GELU(0) should be close to 0
        assert!((gelu_activation(0.0) - 0.0).abs() < 1e-6);

        // GELU should be monotonically increasing for positive x
        let g1 = gelu_activation(1.0);
        let g2 = gelu_activation(2.0);
        assert!(g2 > g1);

        // GELU(-x) should be smaller than GELU(x) for x > 0
        assert!(gelu_activation(-1.0) < gelu_activation(1.0));
    }

    #[test]
    fn test_forward_2d() {
        let config = create_test_config();
        let block = ReversibleBlock::new(&config);

        let x1 = Array2::from_shape_fn((16, 32), |_| rand_normal() * 0.1);
        let x2 = Array2::from_shape_fn((16, 32), |_| rand_normal() * 0.1);

        let (y1, y2) = block.forward_2d(&x1, &x2);

        assert_eq!(y1.dim(), (16, 32));
        assert_eq!(y2.dim(), (16, 32));
    }
}
