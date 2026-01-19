//! Multi-Head Self-Attention implementation

use ndarray::{Array2, Array3, Axis};

/// Multi-Head Self-Attention module
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    d_model: usize,
    num_heads: usize,
    head_dim: usize,
    /// Query projection weights
    w_q: Array2<f64>,
    /// Key projection weights
    w_k: Array2<f64>,
    /// Value projection weights
    w_v: Array2<f64>,
    /// Output projection weights
    w_o: Array2<f64>,
}

impl MultiHeadAttention {
    /// Create new multi-head attention layer
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert!(
            d_model % num_heads == 0,
            "d_model must be divisible by num_heads"
        );

        let head_dim = d_model / num_heads;
        let scale = (1.0 / d_model as f64).sqrt();

        // Initialize weights with Xavier initialization
        let w_q = random_matrix(d_model, d_model, scale);
        let w_k = random_matrix(d_model, d_model, scale);
        let w_v = random_matrix(d_model, d_model, scale);
        let w_o = random_matrix(d_model, d_model, scale);

        Self {
            d_model,
            num_heads,
            head_dim,
            w_q,
            w_k,
            w_v,
            w_o,
        }
    }

    /// Forward pass
    /// Input: (batch, seq_len, d_model)
    /// Output: (batch, seq_len, d_model)
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch, seq_len, _) = x.dim();

        // Linear projections
        let q = self.linear(x, &self.w_q);
        let k = self.linear(x, &self.w_k);
        let v = self.linear(x, &self.w_v);

        // Reshape for multi-head attention: (batch, heads, seq_len, head_dim)
        let q = self.split_heads(&q);
        let k = self.split_heads(&k);
        let v = self.split_heads(&v);

        // Scaled dot-product attention
        let attn_output = self.scaled_dot_product_attention(&q, &k, &v);

        // Merge heads back
        let merged = self.merge_heads(&attn_output);

        // Output projection
        self.linear(&merged, &self.w_o)
    }

    /// Apply linear transformation
    fn linear(&self, x: &Array3<f64>, weight: &Array2<f64>) -> Array3<f64> {
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

    /// Split into multiple heads
    /// (batch, seq_len, d_model) -> (batch, num_heads, seq_len, head_dim)
    fn split_heads(&self, x: &Array3<f64>) -> Vec<Array3<f64>> {
        let (batch, seq_len, _) = x.dim();
        let mut heads = Vec::with_capacity(self.num_heads);

        for h in 0..self.num_heads {
            let mut head = Array3::zeros((batch, seq_len, self.head_dim));
            let start = h * self.head_dim;

            for b in 0..batch {
                for t in 0..seq_len {
                    for d in 0..self.head_dim {
                        head[[b, t, d]] = x[[b, t, start + d]];
                    }
                }
            }
            heads.push(head);
        }

        heads
    }

    /// Merge heads back
    /// (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, d_model)
    fn merge_heads(&self, heads: &[Array3<f64>]) -> Array3<f64> {
        let (batch, seq_len, _) = heads[0].dim();
        let mut output = Array3::zeros((batch, seq_len, self.d_model));

        for (h, head) in heads.iter().enumerate() {
            let start = h * self.head_dim;
            for b in 0..batch {
                for t in 0..seq_len {
                    for d in 0..self.head_dim {
                        output[[b, t, start + d]] = head[[b, t, d]];
                    }
                }
            }
        }

        output
    }

    /// Scaled dot-product attention
    fn scaled_dot_product_attention(
        &self,
        q: &[Array3<f64>],
        k: &[Array3<f64>],
        v: &[Array3<f64>],
    ) -> Vec<Array3<f64>> {
        let scale = (self.head_dim as f64).sqrt();
        let mut outputs = Vec::with_capacity(self.num_heads);

        for h in 0..self.num_heads {
            let (batch, seq_len, _) = q[h].dim();

            // Compute attention scores: Q @ K^T / sqrt(d_k)
            let mut scores = Array3::zeros((batch, seq_len, seq_len));

            for b in 0..batch {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let mut dot = 0.0;
                        for d in 0..self.head_dim {
                            dot += q[h][[b, i, d]] * k[h][[b, j, d]];
                        }
                        scores[[b, i, j]] = dot / scale;
                    }
                }
            }

            // Apply softmax
            let attn_weights = softmax_3d(&scores);

            // Apply attention to values
            let mut output = Array3::zeros((batch, seq_len, self.head_dim));

            for b in 0..batch {
                for i in 0..seq_len {
                    for d in 0..self.head_dim {
                        let mut sum = 0.0;
                        for j in 0..seq_len {
                            sum += attn_weights[[b, i, j]] * v[h][[b, j, d]];
                        }
                        output[[b, i, d]] = sum;
                    }
                }
            }

            outputs.push(output);
        }

        outputs
    }
}

/// Random matrix with Xavier initialization
fn random_matrix(rows: usize, cols: usize, scale: f64) -> Array2<f64> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEED: AtomicU64 = AtomicU64::new(67890);

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

/// Softmax over last dimension of 3D array
fn softmax_3d(x: &Array3<f64>) -> Array3<f64> {
    let (batch, rows, cols) = x.dim();
    let mut output = Array3::zeros((batch, rows, cols));

    for b in 0..batch {
        for i in 0..rows {
            // Find max for numerical stability
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..cols {
                max_val = max_val.max(x[[b, i, j]]);
            }

            // Compute exp and sum
            let mut sum = 0.0;
            for j in 0..cols {
                let exp_val = (x[[b, i, j]] - max_val).exp();
                output[[b, i, j]] = exp_val;
                sum += exp_val;
            }

            // Normalize
            for j in 0..cols {
                output[[b, i, j]] /= sum;
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_head_attention() {
        let mha = MultiHeadAttention::new(64, 4);
        let input = Array3::from_shape_fn((2, 10, 64), |_| 0.1);

        let output = mha.forward(&input);
        assert_eq!(output.dim(), (2, 10, 64));
    }
}
