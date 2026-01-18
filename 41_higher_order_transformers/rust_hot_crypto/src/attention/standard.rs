//! Standard scaled dot-product attention
//!
//! Implements the classic attention mechanism from "Attention Is All You Need"

use ndarray::{Array2, Axis};
use rand::Rng;
use rand_distr::StandardNormal;

/// Standard scaled dot-product attention
#[derive(Debug, Clone)]
pub struct StandardAttention {
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Dimension per head
    pub d_k: usize,
    /// Query projection weights
    pub w_q: Array2<f64>,
    /// Key projection weights
    pub w_k: Array2<f64>,
    /// Value projection weights
    pub w_v: Array2<f64>,
    /// Output projection weights
    pub w_o: Array2<f64>,
}

impl StandardAttention {
    /// Create a new standard attention layer
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        let d_k = d_model / n_heads;
        let scale = (d_model as f64).sqrt();

        let mut rng = rand::thread_rng();

        // Initialize weights with Xavier/Glorot initialization
        let init_weight = |rows: usize, cols: usize| -> Array2<f64> {
            let std = (2.0 / (rows + cols) as f64).sqrt();
            Array2::from_shape_fn((rows, cols), |_| rng.sample::<f64, _>(StandardNormal) * std)
        };

        Self {
            d_model,
            n_heads,
            d_k,
            w_q: init_weight(d_model, d_model),
            w_k: init_weight(d_model, d_model),
            w_v: init_weight(d_model, d_model),
            w_o: init_weight(d_model, d_model),
        }
    }

    /// Forward pass through the attention layer
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape (seq_len, d_model)
    ///
    /// # Returns
    ///
    /// Output tensor of shape (seq_len, d_model)
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let seq_len = x.nrows();

        // Project to Q, K, V
        let q = x.dot(&self.w_q);
        let k = x.dot(&self.w_k);
        let v = x.dot(&self.w_v);

        // Compute attention scores: Q @ K^T / sqrt(d_k)
        let scale = (self.d_k as f64).sqrt();
        let scores = q.dot(&k.t()) / scale;

        // Apply softmax
        let attn_weights = softmax_2d(&scores);

        // Apply attention to values
        let context = attn_weights.dot(&v);

        // Output projection
        context.dot(&self.w_o)
    }

    /// Get attention weights for visualization
    pub fn get_attention_weights(&self, x: &Array2<f64>) -> Array2<f64> {
        let q = x.dot(&self.w_q);
        let k = x.dot(&self.w_k);

        let scale = (self.d_k as f64).sqrt();
        let scores = q.dot(&k.t()) / scale;

        softmax_2d(&scores)
    }
}

/// Apply softmax along the last axis
pub fn softmax_2d(x: &Array2<f64>) -> Array2<f64> {
    let max_vals = x.map_axis(Axis(1), |row| {
        row.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    });

    let exp_x = x.clone() - &max_vals.insert_axis(Axis(1));
    let exp_x = exp_x.mapv(f64::exp);

    let sum_exp = exp_x.sum_axis(Axis(1));

    &exp_x / &sum_exp.insert_axis(Axis(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_standard_attention_shape() {
        let attn = StandardAttention::new(64, 8);
        let x = Array2::zeros((10, 64)); // seq_len=10, d_model=64

        let output = attn.forward(&x);

        assert_eq!(output.shape(), &[10, 64]);
    }

    #[test]
    fn test_softmax() {
        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0]).unwrap();
        let result = softmax_2d(&x);

        // Each row should sum to 1
        for i in 0..result.nrows() {
            let row_sum: f64 = result.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }
}
