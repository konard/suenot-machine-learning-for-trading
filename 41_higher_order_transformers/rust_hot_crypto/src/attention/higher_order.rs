//! Higher-Order Attention with CP Decomposition
//!
//! Extends standard attention to capture third-order (triplet) interactions
//! using tensor decomposition for computational efficiency.

use ndarray::{Array2, Array3, Axis};
use rand::Rng;
use rand_distr::StandardNormal;

use super::standard::softmax_2d;

/// Higher-Order Attention layer
///
/// Instead of computing pairwise attention (Q @ K^T), this computes
/// third-order attention using CP decomposition to approximate
/// the interaction tensor T[i,j,k] = sum_d Q[i,d] * K[j,d] * K[k,d]
#[derive(Debug, Clone)]
pub struct HigherOrderAttention {
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Dimension per head
    pub d_k: usize,
    /// Rank of CP decomposition
    pub rank: usize,
    /// Query projection weights
    pub w_q: Array2<f64>,
    /// Key projection weights
    pub w_k: Array2<f64>,
    /// Value projection weights
    pub w_v: Array2<f64>,
    /// Output projection weights
    pub w_o: Array2<f64>,
    /// CP factor A (for Q)
    pub factor_a: Array2<f64>,
    /// CP factor B (for K, first component)
    pub factor_b: Array2<f64>,
    /// CP factor C (for K, second component)
    pub factor_c: Array2<f64>,
}

impl HigherOrderAttention {
    /// Create a new higher-order attention layer
    ///
    /// # Arguments
    ///
    /// * `d_model` - Model dimension
    /// * `n_heads` - Number of attention heads
    /// * `rank` - Rank of CP decomposition (higher = more expressive, slower)
    pub fn new(d_model: usize, n_heads: usize, rank: usize) -> Self {
        let d_k = d_model / n_heads;

        let mut rng = rand::thread_rng();

        // Xavier/Glorot initialization
        let init_weight = |rows: usize, cols: usize| -> Array2<f64> {
            let std = (2.0 / (rows + cols) as f64).sqrt();
            Array2::from_shape_fn((rows, cols), |_| rng.sample::<f64, _>(StandardNormal) * std)
        };

        // CP factors are smaller matrices
        let init_factor = |rows: usize, cols: usize| -> Array2<f64> {
            let std = (1.0 / cols as f64).sqrt();
            Array2::from_shape_fn((rows, cols), |_| rng.sample::<f64, _>(StandardNormal) * std)
        };

        Self {
            d_model,
            n_heads,
            d_k,
            rank,
            w_q: init_weight(d_model, d_model),
            w_k: init_weight(d_model, d_model),
            w_v: init_weight(d_model, d_model),
            w_o: init_weight(d_model, d_model),
            factor_a: init_factor(d_k, rank),
            factor_b: init_factor(d_k, rank),
            factor_c: init_factor(d_k, rank),
        }
    }

    /// Create with default rank (16)
    pub fn default_rank(d_model: usize, n_heads: usize) -> Self {
        Self::new(d_model, n_heads, 16)
    }

    /// Forward pass through the higher-order attention layer
    ///
    /// The key insight is that instead of computing the full third-order tensor
    /// T[i,j,k], we use CP decomposition:
    ///
    /// T[i,j,k] ≈ sum_r (Q @ a_r)[i] * (K @ b_r)[j] * (K @ c_r)[k]
    ///
    /// This reduces complexity from O(n³) to O(n * r²)
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

        // For simplicity, we'll implement single-head here
        // Project Q and K through CP factors
        let q_a = q.dot(&self.factor_a); // (seq_len, rank)
        let k_b = k.dot(&self.factor_b); // (seq_len, rank)
        let k_c = k.dot(&self.factor_c); // (seq_len, rank)

        // Compute second-order interaction between K components
        // K_bc[j,k] = sum_r K_b[j,r] * K_c[k,r]
        let k_bc = k_b.dot(&k_c.t()); // (seq_len, seq_len)

        // Compute higher-order attention scores
        // For each query position i, we aggregate information from pairs (j,k)
        // We use a simplification: aggregate K_bc along one dimension
        let k_bc_sum = k_bc.sum_axis(Axis(1)); // (seq_len,)

        // Compute attention scores with higher-order component
        let scale = (self.d_k as f64).sqrt();

        // Standard attention scores
        let std_scores = q.dot(&k.t()) / scale;

        // Higher-order component: Q_a contribution weighted by K_bc aggregation
        let ho_scores = q_a.dot(&k_b.t()) * k_bc_sum.insert_axis(Axis(0)) / (scale * self.rank as f64);

        // Combine standard and higher-order scores
        let combined_scores = &std_scores + &ho_scores * 0.3; // Weight the HO component

        // Apply softmax
        let attn_weights = softmax_2d(&combined_scores);

        // Apply attention to values
        let context = attn_weights.dot(&v);

        // Output projection
        context.dot(&self.w_o)
    }

    /// Compute the full third-order attention tensor (for small sequences only!)
    ///
    /// This is O(n³) and should only be used for visualization or debugging
    pub fn compute_full_tensor(&self, x: &Array2<f64>) -> Array3<f64> {
        let seq_len = x.nrows();

        let q = x.dot(&self.w_q);
        let k = x.dot(&self.w_k);

        let q_a = q.dot(&self.factor_a);
        let k_b = k.dot(&self.factor_b);
        let k_c = k.dot(&self.factor_c);

        // Build full tensor T[i,j,k] = sum_r Q_a[i,r] * K_b[j,r] * K_c[k,r]
        let mut tensor = Array3::zeros((seq_len, seq_len, seq_len));

        for i in 0..seq_len {
            for j in 0..seq_len {
                for k in 0..seq_len {
                    let mut val = 0.0;
                    for r in 0..self.rank {
                        val += q_a[[i, r]] * k_b[[j, r]] * k_c[[k, r]];
                    }
                    tensor[[i, j, k]] = val;
                }
            }
        }

        tensor
    }

    /// Get attention weights for the first-order component (for visualization)
    pub fn get_attention_weights(&self, x: &Array2<f64>) -> Array2<f64> {
        let q = x.dot(&self.w_q);
        let k = x.dot(&self.w_k);

        let q_a = q.dot(&self.factor_a);
        let k_b = k.dot(&self.factor_b);
        let k_c = k.dot(&self.factor_c);

        let k_bc = k_b.dot(&k_c.t());
        let k_bc_sum = k_bc.sum_axis(Axis(1));

        let scale = (self.d_k as f64).sqrt();
        let std_scores = q.dot(&k.t()) / scale;
        let ho_scores = q_a.dot(&k_b.t()) * k_bc_sum.insert_axis(Axis(0)) / (scale * self.rank as f64);

        let combined_scores = &std_scores + &ho_scores * 0.3;

        softmax_2d(&combined_scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_higher_order_attention_shape() {
        let attn = HigherOrderAttention::new(64, 8, 16);
        let x = Array2::zeros((10, 64)); // seq_len=10, d_model=64

        let output = attn.forward(&x);

        assert_eq!(output.shape(), &[10, 64]);
    }

    #[test]
    fn test_full_tensor_shape() {
        let attn = HigherOrderAttention::new(32, 4, 8);
        let x = Array2::from_shape_fn((5, 32), |_| rand::random::<f64>());

        let tensor = attn.compute_full_tensor(&x);

        assert_eq!(tensor.shape(), &[5, 5, 5]);
    }

    #[test]
    fn test_attention_weights_sum_to_one() {
        let attn = HigherOrderAttention::new(32, 4, 8);
        let x = Array2::from_shape_fn((5, 32), |_| rand::random::<f64>());

        let weights = attn.get_attention_weights(&x);

        for i in 0..weights.nrows() {
            let row_sum: f64 = weights.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }
}
