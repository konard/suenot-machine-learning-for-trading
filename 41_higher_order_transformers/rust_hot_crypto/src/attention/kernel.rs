//! Kernel Attention for Linear Complexity
//!
//! Uses kernel feature maps to achieve O(n) complexity instead of O(n²)

use ndarray::{Array2, Axis};
use rand::Rng;
use rand_distr::StandardNormal;

/// Kernel attention with exponential feature map
///
/// Instead of computing softmax(Q @ K^T) @ V explicitly,
/// we use kernel approximation: φ(Q) @ (φ(K)^T @ V) / normalizer
///
/// This achieves O(nd²) complexity instead of O(n²d)
#[derive(Debug, Clone)]
pub struct KernelAttention {
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Dimension per head
    pub d_k: usize,
    /// Number of random features for kernel approximation
    pub num_features: usize,
    /// Query projection weights
    pub w_q: Array2<f64>,
    /// Key projection weights
    pub w_k: Array2<f64>,
    /// Value projection weights
    pub w_v: Array2<f64>,
    /// Output projection weights
    pub w_o: Array2<f64>,
    /// Random features for kernel approximation
    pub random_features: Array2<f64>,
}

impl KernelAttention {
    /// Create a new kernel attention layer
    ///
    /// # Arguments
    ///
    /// * `d_model` - Model dimension
    /// * `n_heads` - Number of attention heads
    /// * `num_features` - Number of random features (higher = better approximation)
    pub fn new(d_model: usize, n_heads: usize, num_features: usize) -> Self {
        let d_k = d_model / n_heads;

        let mut rng = rand::thread_rng();

        let init_weight = |rows: usize, cols: usize| -> Array2<f64> {
            let std = (2.0 / (rows + cols) as f64).sqrt();
            Array2::from_shape_fn((rows, cols), |_| rng.sample::<f64, _>(StandardNormal) * std)
        };

        // Random features for kernel approximation (Gaussian kernel)
        let random_features = Array2::from_shape_fn((d_model, num_features), |_| {
            rng.sample::<f64, _>(StandardNormal)
        });

        Self {
            d_model,
            n_heads,
            d_k,
            num_features,
            w_q: init_weight(d_model, d_model),
            w_k: init_weight(d_model, d_model),
            w_v: init_weight(d_model, d_model),
            w_o: init_weight(d_model, d_model),
            random_features,
        }
    }

    /// Create with default number of features (64)
    pub fn default_features(d_model: usize, n_heads: usize) -> Self {
        Self::new(d_model, n_heads, 64)
    }

    /// Apply the kernel feature map φ(x) = exp(x) / ||exp(x)||
    ///
    /// Uses random Fourier features for approximating softmax kernel
    fn feature_map(&self, x: &Array2<f64>) -> Array2<f64> {
        // Project through random features
        let projected = x.dot(&self.random_features);

        // Apply ELU + 1 as feature map (positive features)
        let features = projected.mapv(|v| {
            if v > 0.0 {
                v + 1.0
            } else {
                v.exp()
            }
        });

        // Normalize
        let norms = features.map_axis(Axis(1), |row| {
            let sum_sq: f64 = row.iter().map(|x| x * x).sum();
            sum_sq.sqrt().max(1e-10)
        });

        &features / &norms.insert_axis(Axis(1))
    }

    /// Forward pass through kernel attention
    ///
    /// Uses the identity: softmax(Q @ K^T) @ V ≈ φ(Q) @ (φ(K)^T @ V) / Z
    /// where Z is the normalizing constant
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape (seq_len, d_model)
    ///
    /// # Returns
    ///
    /// Output tensor of shape (seq_len, d_model)
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        // Project to Q, K, V
        let q = x.dot(&self.w_q);
        let k = x.dot(&self.w_k);
        let v = x.dot(&self.w_v);

        // Apply feature maps
        let phi_q = self.feature_map(&q);
        let phi_k = self.feature_map(&k);

        // Compute kernel attention
        // KV = φ(K)^T @ V  -- shape: (num_features, d_model)
        let kv = phi_k.t().dot(&v);

        // K_sum = φ(K)^T @ 1  -- shape: (num_features,) for normalization
        let k_sum: Array2<f64> = phi_k.t().dot(&Array2::ones((phi_k.nrows(), 1)));

        // Output = φ(Q) @ KV  -- shape: (seq_len, d_model)
        let numerator = phi_q.dot(&kv);

        // Normalizer = φ(Q) @ K_sum  -- shape: (seq_len, 1)
        let normalizer = phi_q.dot(&k_sum);

        // Normalize
        let context = &numerator / &normalizer;

        // Output projection
        context.dot(&self.w_o)
    }

    /// Estimate attention weights (approximate, for visualization)
    ///
    /// Note: This computes the full O(n²) attention for visualization only
    pub fn estimate_attention_weights(&self, x: &Array2<f64>) -> Array2<f64> {
        let q = x.dot(&self.w_q);
        let k = x.dot(&self.w_k);

        let phi_q = self.feature_map(&q);
        let phi_k = self.feature_map(&k);

        // Approximate attention weights
        let scores = phi_q.dot(&phi_k.t());

        // Normalize rows
        let row_sums = scores.sum_axis(Axis(1));
        &scores / &row_sums.insert_axis(Axis(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_attention_shape() {
        let attn = KernelAttention::new(64, 8, 32);
        let x = Array2::zeros((100, 64)); // seq_len=100, d_model=64

        let output = attn.forward(&x);

        assert_eq!(output.shape(), &[100, 64]);
    }

    #[test]
    fn test_feature_map_normalization() {
        let attn = KernelAttention::new(32, 4, 16);
        let x = Array2::from_shape_fn((10, 32), |_| rand::random::<f64>());

        let features = attn.feature_map(&x);

        // Each row should have unit norm (approximately)
        for i in 0..features.nrows() {
            let norm: f64 = features.row(i).iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_attention_weights_positive() {
        let attn = KernelAttention::new(32, 4, 16);
        let x = Array2::from_shape_fn((5, 32), |_| rand::random::<f64>());

        let weights = attn.estimate_attention_weights(&x);

        // All weights should be non-negative
        for w in weights.iter() {
            assert!(*w >= 0.0);
        }
    }
}
