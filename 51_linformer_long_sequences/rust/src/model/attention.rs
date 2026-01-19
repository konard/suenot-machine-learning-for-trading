//! Linformer attention implementation.
//!
//! Implements efficient O(n) attention using low-rank projection.

use ndarray::{Array2, Array3, Axis, s};
use rand::Rng;

/// Linformer attention layer with linear complexity.
///
/// Instead of computing full n×n attention matrix, projects keys and values
/// from n to k dimensions, achieving O(nk) complexity where k << n.
pub struct LinformerAttention {
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Sequence length
    pub seq_len: usize,
    /// Projection dimension
    pub k: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Query projection weights [n_heads, d_model, head_dim]
    pub w_q: Array3<f64>,
    /// Key projection weights [n_heads, d_model, head_dim]
    pub w_k: Array3<f64>,
    /// Value projection weights [n_heads, d_model, head_dim]
    pub w_v: Array3<f64>,
    /// Output projection weights [d_model, d_model]
    pub w_o: Array2<f64>,
    /// Key projection matrix E [n_heads, k, seq_len]
    pub e_proj: Array3<f64>,
    /// Value projection matrix F [n_heads, k, seq_len]
    pub f_proj: Array3<f64>,
    /// Whether E and F are shared
    pub share_kv: bool,
}

impl LinformerAttention {
    /// Create a new Linformer attention layer.
    pub fn new(d_model: usize, n_heads: usize, seq_len: usize, k: usize, share_kv: bool) -> Self {
        assert!(d_model % n_heads == 0, "d_model must be divisible by n_heads");

        let head_dim = d_model / n_heads;
        let mut rng = rand::thread_rng();

        // Xavier initialization scale
        let scale_qkv = (2.0 / (d_model + head_dim) as f64).sqrt();
        let scale_o = (2.0 / (d_model * 2) as f64).sqrt();
        let scale_proj = (2.0 / (k + seq_len) as f64).sqrt();

        // Initialize projection weights
        let w_q = Array3::from_shape_fn((n_heads, d_model, head_dim), |_| {
            rng.gen::<f64>() * scale_qkv * 2.0 - scale_qkv
        });
        let w_k = Array3::from_shape_fn((n_heads, d_model, head_dim), |_| {
            rng.gen::<f64>() * scale_qkv * 2.0 - scale_qkv
        });
        let w_v = Array3::from_shape_fn((n_heads, d_model, head_dim), |_| {
            rng.gen::<f64>() * scale_qkv * 2.0 - scale_qkv
        });
        let w_o = Array2::from_shape_fn((d_model, d_model), |_| {
            rng.gen::<f64>() * scale_o * 2.0 - scale_o
        });

        // Initialize projection matrices E and F
        let e_proj = Array3::from_shape_fn((n_heads, k, seq_len), |_| {
            rng.gen::<f64>() * scale_proj * 2.0 - scale_proj
        });

        let f_proj = if share_kv {
            e_proj.clone()
        } else {
            Array3::from_shape_fn((n_heads, k, seq_len), |_| {
                rng.gen::<f64>() * scale_proj * 2.0 - scale_proj
            })
        };

        Self {
            d_model,
            n_heads,
            seq_len,
            k,
            head_dim,
            w_q,
            w_k,
            w_v,
            w_o,
            e_proj,
            f_proj,
            share_kv,
        }
    }

    /// Compute attention for a single sequence.
    /// Input: [seq_len, d_model]
    /// Output: [seq_len, d_model]
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let (seq_len, _) = x.dim();
        let actual_seq_len = seq_len.min(self.seq_len);

        // Prepare output accumulator
        let mut output = Array2::zeros((actual_seq_len, self.d_model));

        // Process each head
        for h in 0..self.n_heads {
            // Project to Q, K, V
            let q = self.project_qkv(x, &self.w_q.slice(s![h, .., ..]).to_owned());
            let k = self.project_qkv(x, &self.w_k.slice(s![h, .., ..]).to_owned());
            let v = self.project_qkv(x, &self.w_v.slice(s![h, .., ..]).to_owned());

            // Get projection matrices for this head
            let e = self.e_proj.slice(s![h, .., ..actual_seq_len]).to_owned();
            let f = self.f_proj.slice(s![h, .., ..actual_seq_len]).to_owned();

            // Project keys and values: E*K, F*V (reduce from n to k)
            let k_proj = self.matmul(&e, &k); // [k, head_dim]
            let v_proj = self.matmul(&f, &v); // [k, head_dim]

            // Compute attention scores: Q * (E*K)^T / sqrt(d_k)
            let scale = (self.head_dim as f64).sqrt();
            let scores = self.matmul(&q, &k_proj.t().to_owned()); // [seq_len, k]

            // Scale and softmax
            let scaled_scores = &scores / scale;
            let attention_weights = self.softmax(&scaled_scores);

            // Apply attention: softmax(scores) * (F*V)
            let head_output = self.matmul(&attention_weights, &v_proj); // [seq_len, head_dim]

            // Accumulate to output
            let start = h * self.head_dim;
            for i in 0..actual_seq_len {
                for j in 0..self.head_dim {
                    output[[i, start + j]] = head_output[[i, j]];
                }
            }
        }

        // Output projection
        self.matmul(&output, &self.w_o)
    }

    /// Project input through weight matrix.
    fn project_qkv(&self, x: &Array2<f64>, w: &Array2<f64>) -> Array2<f64> {
        self.matmul(x, w)
    }

    /// Matrix multiplication helper.
    fn matmul(&self, a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        a.dot(b)
    }

    /// Softmax along the last axis.
    fn softmax(&self, x: &Array2<f64>) -> Array2<f64> {
        let (rows, cols) = x.dim();
        let mut result = Array2::zeros((rows, cols));

        for i in 0..rows {
            let row = x.row(i);
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_vals: Vec<f64> = row.iter().map(|&v| (v - max_val).exp()).collect();
            let sum: f64 = exp_vals.iter().sum();

            for (j, &exp_val) in exp_vals.iter().enumerate() {
                result[[i, j]] = exp_val / sum;
            }
        }

        result
    }

    /// Get the memory complexity description.
    pub fn memory_complexity(&self) -> String {
        let standard = self.seq_len * self.seq_len;
        let linformer = self.seq_len * self.k;
        let reduction = 100.0 * (1.0 - linformer as f64 / standard as f64);

        format!(
            "Standard: O(n²) = {} | Linformer: O(nk) = {} | Reduction: {:.1}%",
            standard, linformer, reduction
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_creation() {
        let attn = LinformerAttention::new(64, 4, 128, 32, true);
        assert_eq!(attn.d_model, 64);
        assert_eq!(attn.n_heads, 4);
        assert_eq!(attn.head_dim, 16);
    }

    #[test]
    fn test_attention_forward() {
        let attn = LinformerAttention::new(32, 2, 64, 16, true);
        let input = Array2::from_shape_fn((64, 32), |(i, j)| {
            ((i * 32 + j) as f64 * 0.01).sin()
        });

        let output = attn.forward(&input);
        assert_eq!(output.dim(), (64, 32));
    }

    #[test]
    fn test_memory_complexity() {
        let attn = LinformerAttention::new(128, 4, 512, 64, true);
        let complexity = attn.memory_complexity();
        assert!(complexity.contains("Reduction"));
    }

    #[test]
    fn test_softmax() {
        let attn = LinformerAttention::new(32, 2, 32, 16, true);
        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0]).unwrap();
        let result = attn.softmax(&x);

        // Each row should sum to 1
        for i in 0..2 {
            let sum: f64 = result.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}
