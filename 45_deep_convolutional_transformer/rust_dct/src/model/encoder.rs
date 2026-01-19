//! Transformer Encoder Layer

use super::attention::MultiHeadAttention;
use ndarray::{Array2, Array3};

/// Feed-forward network with separable structure
#[derive(Debug, Clone)]
pub struct SeparableFeedForward {
    d_model: usize,
    d_ff: usize,
    /// First linear layer weights
    w1: Array2<f64>,
    /// Second linear layer weights
    w2: Array2<f64>,
}

impl SeparableFeedForward {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let scale1 = (2.0 / (d_model + d_ff) as f64).sqrt();
        let scale2 = (2.0 / (d_ff + d_model) as f64).sqrt();

        Self {
            d_model,
            d_ff,
            w1: random_matrix(d_model, d_ff, scale1),
            w2: random_matrix(d_ff, d_model, scale2),
        }
    }

    /// Forward pass with GELU activation
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let hidden = linear_3d(x, &self.w1);
        let activated = gelu(&hidden);
        linear_3d(&activated, &self.w2)
    }
}

/// Transformer encoder layer
#[derive(Debug, Clone)]
pub struct TransformerEncoderLayer {
    attention: MultiHeadAttention,
    ffn: SeparableFeedForward,
    /// Layer norm parameters
    ln1_gamma: Vec<f64>,
    ln1_beta: Vec<f64>,
    ln2_gamma: Vec<f64>,
    ln2_beta: Vec<f64>,
}

impl TransformerEncoderLayer {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        Self {
            attention: MultiHeadAttention::new(d_model, num_heads),
            ffn: SeparableFeedForward::new(d_model, d_ff),
            ln1_gamma: vec![1.0; d_model],
            ln1_beta: vec![0.0; d_model],
            ln2_gamma: vec![1.0; d_model],
            ln2_beta: vec![0.0; d_model],
        }
    }

    /// Forward pass with residual connections and layer norm
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        // Self-attention with residual and layer norm
        let attn_output = self.attention.forward(x);
        let x1 = add_tensors(x, &attn_output);
        let x1 = layer_norm(&x1, &self.ln1_gamma, &self.ln1_beta);

        // Feed-forward with residual and layer norm
        let ffn_output = self.ffn.forward(&x1);
        let x2 = add_tensors(&x1, &ffn_output);
        layer_norm(&x2, &self.ln2_gamma, &self.ln2_beta)
    }
}

/// Transformer encoder with multiple layers
#[derive(Debug, Clone)]
pub struct TransformerEncoder {
    layers: Vec<TransformerEncoderLayer>,
}

impl TransformerEncoder {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, num_layers: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| TransformerEncoderLayer::new(d_model, num_heads, d_ff))
            .collect();

        Self { layers }
    }

    /// Forward pass through all encoder layers
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let mut output = x.clone();

        for layer in &self.layers {
            output = layer.forward(&output);
        }

        output
    }
}

/// Linear transformation for 3D tensor
fn linear_3d(x: &Array3<f64>, weight: &Array2<f64>) -> Array3<f64> {
    let (batch, seq_len, in_dim) = x.dim();
    let out_dim = weight.ncols();

    let mut output = Array3::zeros((batch, seq_len, out_dim));

    for b in 0..batch {
        for t in 0..seq_len {
            for o in 0..out_dim {
                let mut sum = 0.0;
                for i in 0..in_dim {
                    sum += x[[b, t, i]] * weight[[i, o]];
                }
                output[[b, t, o]] = sum;
            }
        }
    }

    output
}

/// GELU activation function
fn gelu(x: &Array3<f64>) -> Array3<f64> {
    x.mapv(|v| {
        0.5 * v * (1.0 + (std::f64::consts::FRAC_2_SQRT_PI * (v + 0.044715 * v.powi(3))).tanh())
    })
}

/// Layer normalization
fn layer_norm(x: &Array3<f64>, gamma: &[f64], beta: &[f64]) -> Array3<f64> {
    let (batch, seq_len, d_model) = x.dim();
    let mut output = Array3::zeros((batch, seq_len, d_model));
    let eps = 1e-5;

    for b in 0..batch {
        for t in 0..seq_len {
            // Compute mean
            let mut mean = 0.0;
            for d in 0..d_model {
                mean += x[[b, t, d]];
            }
            mean /= d_model as f64;

            // Compute variance
            let mut var = 0.0;
            for d in 0..d_model {
                var += (x[[b, t, d]] - mean).powi(2);
            }
            var /= d_model as f64;

            // Normalize
            let std = (var + eps).sqrt();
            for d in 0..d_model {
                output[[b, t, d]] = gamma[d] * (x[[b, t, d]] - mean) / std + beta[d];
            }
        }
    }

    output
}

/// Element-wise addition
fn add_tensors(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
    a + b
}

/// Random matrix initialization
fn random_matrix(rows: usize, cols: usize, scale: f64) -> Array2<f64> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEED: AtomicU64 = AtomicU64::new(11111);

    Array2::from_shape_fn((rows, cols), |_| {
        let s = SEED.fetch_add(1, Ordering::Relaxed);
        let u1 = ((s.wrapping_mul(1103515245).wrapping_add(12345) % (1 << 31)) as f64)
            / (1u64 << 31) as f64;
        let u2 = ((s.wrapping_mul(1103515245).wrapping_add(54321) % (1 << 31)) as f64)
            / (1u64 << 31) as f64;

        let u1 = u1.max(1e-10);
        let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        normal * scale
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_encoder() {
        let encoder = TransformerEncoder::new(64, 4, 256, 2);
        let input = Array3::from_shape_fn((2, 10, 64), |_| 0.1);

        let output = encoder.forward(&input);
        assert_eq!(output.dim(), (2, 10, 64));
    }
}
