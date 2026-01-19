//! LSH (Locality-Sensitive Hashing) Attention
//!
//! Implements efficient O(L·log(L)) attention using locality-sensitive hashing.
//! Similar vectors are hashed to the same bucket, allowing attention to be
//! computed only within buckets.

use ndarray::{Array1, Array2, Array3, Array4, Axis, s};
use std::cmp::Ordering;
use std::f64::consts::PI;

use super::config::ReformerConfig;

/// Attention weights for interpretation
#[derive(Debug, Clone)]
pub struct AttentionWeights {
    /// Temporal attention weights [batch, n_heads, seq_len, seq_len]
    pub temporal_weights: Option<Array4<f64>>,
    /// Hash bucket assignments [batch, n_heads, seq_len]
    pub bucket_assignments: Option<Array3<i32>>,
}

impl AttentionWeights {
    pub fn new() -> Self {
        Self {
            temporal_weights: None,
            bucket_assignments: None,
        }
    }
}

impl Default for AttentionWeights {
    fn default() -> Self {
        Self::new()
    }
}

/// LSH Attention layer
///
/// Uses locality-sensitive hashing to reduce attention complexity from O(L²) to O(L·log(L))
#[derive(Debug, Clone)]
pub struct LSHAttention {
    /// Query projection [d_model, d_model]
    w_q: Array2<f64>,
    /// Key projection [d_model, d_model]
    w_k: Array2<f64>,
    /// Value projection [d_model, d_model]
    w_v: Array2<f64>,
    /// Output projection [d_model, d_model]
    w_o: Array2<f64>,
    /// Random rotation matrices for hashing [n_rounds, head_dim, n_buckets/2]
    random_rotations: Vec<Array2<f64>>,
    /// Number of attention heads
    n_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Number of hash rounds
    n_rounds: usize,
    /// Number of buckets
    n_buckets: usize,
    /// Chunk size for attention
    chunk_size: usize,
    /// Scaling factor
    scale: f64,
    /// Whether to use causal masking
    causal: bool,
}

impl LSHAttention {
    /// Create a new LSH attention layer
    pub fn new(config: &ReformerConfig) -> Self {
        let d_model = config.d_model;
        let n_heads = config.n_heads;
        let head_dim = d_model / n_heads;
        let n_rounds = config.n_hash_rounds;
        let n_buckets = config.n_buckets;

        // Xavier initialization
        let scale_init = (2.0 / (d_model * 2) as f64).sqrt();

        let w_q = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale_init);
        let w_k = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale_init);
        let w_v = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale_init);
        let w_o = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale_init);

        // Create random rotation matrices for LSH
        let random_rotations = (0..n_rounds)
            .map(|_| {
                Array2::from_shape_fn((head_dim, n_buckets / 2), |_| rand_normal())
            })
            .collect();

        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            random_rotations,
            n_heads,
            head_dim,
            n_rounds,
            n_buckets,
            chunk_size: config.chunk_size,
            scale: (head_dim as f64).sqrt(),
            causal: config.causal,
        }
    }

    /// Forward pass with LSH attention
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, d_model]
    ///
    /// # Returns
    /// * Output tensor [batch, seq_len, d_model]
    /// * Attention weights for interpretation
    pub fn forward(&self, x: &Array3<f64>) -> (Array3<f64>, AttentionWeights) {
        let (batch_size, seq_len, d_model) = x.dim();

        // Linear projections
        let q = self.linear_transform_3d(x, &self.w_q);
        let k = self.linear_transform_3d(x, &self.w_k);
        let v = self.linear_transform_3d(x, &self.w_v);

        let mut output = Array3::zeros((batch_size, seq_len, d_model));
        let mut all_buckets = Array3::zeros((batch_size, self.n_heads, seq_len));

        for b in 0..batch_size {
            for h in 0..self.n_heads {
                let h_start = h * self.head_dim;
                let h_end = (h + 1) * self.head_dim;

                // Extract head-specific Q, K, V
                let q_head = extract_head(&q, b, h_start, h_end);
                let k_head = extract_head(&k, b, h_start, h_end);
                let v_head = extract_head(&v, b, h_start, h_end);

                // Compute LSH attention for this head
                let (head_output, buckets) =
                    self.lsh_attention_head(&q_head, &k_head, &v_head);

                // Store output
                for t in 0..seq_len {
                    for d in 0..self.head_dim {
                        output[[b, t, h_start + d]] = head_output[[t, d]];
                    }
                    all_buckets[[b, h, t]] = buckets[t] as f64;
                }
            }
        }

        // Output projection
        let output = self.linear_transform_3d(&output, &self.w_o);

        let weights = AttentionWeights {
            temporal_weights: None,
            bucket_assignments: Some(all_buckets.mapv(|x| x as i32)),
        };

        (output, weights)
    }

    /// Compute LSH attention for a single head
    fn lsh_attention_head(
        &self,
        q: &Array2<f64>,
        k: &Array2<f64>,
        v: &Array2<f64>,
    ) -> (Array2<f64>, Vec<i32>) {
        let seq_len = q.nrows();

        // For short sequences, use full attention
        if seq_len <= self.chunk_size * 2 {
            return self.full_attention(q, k, v);
        }

        // Multi-round hashing
        let mut attention_counts = vec![0.0; seq_len];
        let mut output = Array2::zeros((seq_len, self.head_dim));
        let mut final_buckets = vec![0i32; seq_len];

        for round in 0..self.n_rounds {
            // Hash Q and K
            let q_hashes = self.hash_vectors(q, round);
            let k_hashes = self.hash_vectors(k, round);

            // Store bucket assignments from first round
            if round == 0 {
                final_buckets = q_hashes.iter().map(|&h| h as i32).collect();
            }

            // Group by bucket and compute attention within buckets
            let (round_output, counts) = self.bucket_attention(q, k, v, &q_hashes, &k_hashes);

            // Accumulate
            for i in 0..seq_len {
                if counts[i] > 0.0 {
                    for d in 0..self.head_dim {
                        output[[i, d]] += round_output[[i, d]];
                    }
                    attention_counts[i] += counts[i];
                }
            }
        }

        // Average across rounds
        for i in 0..seq_len {
            if attention_counts[i] > 0.0 {
                for d in 0..self.head_dim {
                    output[[i, d]] /= attention_counts[i];
                }
            }
        }

        (output, final_buckets)
    }

    /// Hash vectors using random rotation projection
    fn hash_vectors(&self, vectors: &Array2<f64>, round: usize) -> Vec<usize> {
        let rotation = &self.random_rotations[round];
        let seq_len = vectors.nrows();

        let mut hashes = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            // Project through rotation matrix
            let mut projected = vec![0.0; self.n_buckets / 2];
            for j in 0..self.n_buckets / 2 {
                let mut sum = 0.0;
                for d in 0..self.head_dim {
                    sum += vectors[[i, d]] * rotation[[d, j]];
                }
                projected[j] = sum;
            }

            // Angular LSH: concatenate with negation and find argmax
            let mut full = Vec::with_capacity(self.n_buckets);
            full.extend(&projected);
            for &p in &projected {
                full.push(-p);
            }

            // Hash = argmax
            let hash = full
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            hashes.push(hash);
        }

        hashes
    }

    /// Compute attention within buckets
    fn bucket_attention(
        &self,
        q: &Array2<f64>,
        k: &Array2<f64>,
        v: &Array2<f64>,
        q_hashes: &[usize],
        k_hashes: &[usize],
    ) -> (Array2<f64>, Vec<f64>) {
        let seq_len = q.nrows();

        // Group indices by bucket
        let mut buckets: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();

        for (i, &hash) in k_hashes.iter().enumerate() {
            buckets.entry(hash).or_insert_with(Vec::new).push(i);
        }

        let mut output = Array2::zeros((seq_len, self.head_dim));
        let mut counts = vec![0.0; seq_len];

        // For each query, attend only to keys in the same bucket
        for (i, &q_hash) in q_hashes.iter().enumerate() {
            if let Some(bucket_indices) = buckets.get(&q_hash) {
                let bucket_size = bucket_indices.len();
                if bucket_size == 0 {
                    continue;
                }

                // Compute attention scores within bucket
                let mut scores = Vec::with_capacity(bucket_size);
                for &j in bucket_indices {
                    // Skip future positions if causal
                    if self.causal && j > i {
                        scores.push(f64::NEG_INFINITY);
                        continue;
                    }

                    let mut score = 0.0;
                    for d in 0..self.head_dim {
                        score += q[[i, d]] * k[[j, d]];
                    }
                    scores.push(score / self.scale);
                }

                // Softmax
                let max_score = scores
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);

                if max_score == f64::NEG_INFINITY {
                    continue;
                }

                let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
                let sum_exp: f64 = exp_scores.iter().sum();

                if sum_exp == 0.0 {
                    continue;
                }

                let attention_probs: Vec<f64> = exp_scores.iter().map(|e| e / sum_exp).collect();

                // Compute weighted sum of values
                for (k_idx, (&j, &prob)) in bucket_indices.iter().zip(attention_probs.iter()).enumerate()
                {
                    if prob > 0.0 {
                        for d in 0..self.head_dim {
                            output[[i, d]] += prob * v[[j, d]];
                        }
                    }
                }

                counts[i] = 1.0;
            }
        }

        (output, counts)
    }

    /// Full attention (for short sequences or fallback)
    fn full_attention(
        &self,
        q: &Array2<f64>,
        k: &Array2<f64>,
        v: &Array2<f64>,
    ) -> (Array2<f64>, Vec<i32>) {
        let seq_len = q.nrows();

        let mut output = Array2::zeros((seq_len, self.head_dim));
        let buckets = vec![0i32; seq_len];

        for i in 0..seq_len {
            // Compute attention scores
            let mut scores = Vec::with_capacity(seq_len);

            for j in 0..seq_len {
                if self.causal && j > i {
                    scores.push(f64::NEG_INFINITY);
                } else {
                    let mut score = 0.0;
                    for d in 0..self.head_dim {
                        score += q[[i, d]] * k[[j, d]];
                    }
                    scores.push(score / self.scale);
                }
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exp_scores.iter().sum();

            // Weighted sum
            for j in 0..seq_len {
                let prob = exp_scores[j] / sum_exp;
                for d in 0..self.head_dim {
                    output[[i, d]] += prob * v[[j, d]];
                }
            }
        }

        (output, buckets)
    }

    /// Linear transformation for 3D tensor
    fn linear_transform_3d(&self, x: &Array3<f64>, w: &Array2<f64>) -> Array3<f64> {
        let (batch_size, seq_len, d_in) = x.dim();
        let d_out = w.ncols();

        let mut output = Array3::zeros((batch_size, seq_len, d_out));

        for b in 0..batch_size {
            for t in 0..seq_len {
                for d_o in 0..d_out {
                    let mut sum = 0.0;
                    for d_i in 0..d_in {
                        sum += x[[b, t, d_i]] * w[[d_i, d_o]];
                    }
                    output[[b, t, d_o]] = sum;
                }
            }
        }

        output
    }

    /// Forward pass for 2D input [seq_len, d_model]
    pub fn forward_2d(&self, x: &Array2<f64>) -> (Array2<f64>, Vec<i32>) {
        let (seq_len, d_model) = x.dim();

        // Add batch dimension
        let x_3d = x.clone().insert_axis(Axis(0));

        let (output_3d, weights) = self.forward(&x_3d);

        // Remove batch dimension
        let output = output_3d.index_axis(Axis(0), 0).to_owned();

        let buckets = weights
            .bucket_assignments
            .map(|b| {
                let head_buckets = b.index_axis(Axis(0), 0);
                let first_head = head_buckets.index_axis(Axis(0), 0);
                first_head.iter().cloned().collect()
            })
            .unwrap_or_else(|| vec![0; seq_len]);

        (output, buckets)
    }
}

/// Extract head-specific slice from projected tensor
fn extract_head(tensor: &Array3<f64>, batch: usize, h_start: usize, h_end: usize) -> Array2<f64> {
    let seq_len = tensor.shape()[1];
    let head_dim = h_end - h_start;

    let mut head = Array2::zeros((seq_len, head_dim));

    for t in 0..seq_len {
        for d in 0..head_dim {
            head[[t, d]] = tensor[[batch, t, h_start + d]];
        }
    }

    head
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
            n_heads: 4,
            n_hash_rounds: 2,
            n_buckets: 8,
            chunk_size: 8,
            seq_len: 32,
            causal: false,
            ..Default::default()
        }
    }

    #[test]
    fn test_lsh_attention_creation() {
        let config = create_test_config();
        let attention = LSHAttention::new(&config);

        assert_eq!(attention.n_heads, 4);
        assert_eq!(attention.head_dim, 8);
        assert_eq!(attention.n_rounds, 2);
        assert_eq!(attention.n_buckets, 8);
    }

    #[test]
    fn test_forward_3d() {
        let config = create_test_config();
        let attention = LSHAttention::new(&config);

        let x = Array3::from_shape_fn((2, 16, 32), |_| rand_normal());
        let (output, weights) = attention.forward(&x);

        assert_eq!(output.dim(), (2, 16, 32));
        assert!(weights.bucket_assignments.is_some());
    }

    #[test]
    fn test_forward_2d() {
        let config = create_test_config();
        let attention = LSHAttention::new(&config);

        let x = Array2::from_shape_fn((16, 32), |_| rand_normal());
        let (output, buckets) = attention.forward_2d(&x);

        assert_eq!(output.dim(), (16, 32));
        assert_eq!(buckets.len(), 16);
    }

    #[test]
    fn test_hash_vectors() {
        let config = create_test_config();
        let attention = LSHAttention::new(&config);

        let vectors = Array2::from_shape_fn((10, 8), |_| rand_normal());
        let hashes = attention.hash_vectors(&vectors, 0);

        assert_eq!(hashes.len(), 10);

        // All hashes should be valid bucket indices
        for &h in &hashes {
            assert!(h < config.n_buckets);
        }
    }

    #[test]
    fn test_similar_vectors_same_bucket() {
        let config = ReformerConfig {
            d_model: 32,
            n_heads: 4,
            n_hash_rounds: 8, // More rounds for better accuracy
            n_buckets: 8,
            ..Default::default()
        };
        let attention = LSHAttention::new(&config);

        // Create two similar vectors
        let base = Array1::from_shape_fn(8, |_| rand_normal());
        let similar = &base + &Array1::from_shape_fn(8, |_| rand_normal() * 0.01);
        let different = Array1::from_shape_fn(8, |_| rand_normal());

        let mut vectors = Array2::zeros((3, 8));
        for d in 0..8 {
            vectors[[0, d]] = base[d];
            vectors[[1, d]] = similar[d];
            vectors[[2, d]] = different[d];
        }

        // Count matching buckets across rounds
        let mut same_bucket_count = 0;
        for round in 0..config.n_hash_rounds {
            let hashes = attention.hash_vectors(&vectors, round);
            if hashes[0] == hashes[1] {
                same_bucket_count += 1;
            }
        }

        // Similar vectors should often hash to the same bucket
        // (probabilistic, so we just check it happens at least sometimes)
        assert!(
            same_bucket_count > 0,
            "Similar vectors should share buckets in at least one round"
        );
    }

    #[test]
    fn test_causal_masking() {
        let mut config = create_test_config();
        config.causal = true;
        let attention = LSHAttention::new(&config);

        let x = Array3::from_shape_fn((1, 8, 32), |_| rand_normal());
        let (output, _) = attention.forward(&x);

        // Output should be valid (not NaN or Inf)
        for &val in output.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_long_sequence() {
        let config = ReformerConfig {
            d_model: 32,
            n_heads: 4,
            n_hash_rounds: 4,
            n_buckets: 16,
            chunk_size: 32,
            seq_len: 256,
            ..Default::default()
        };
        let attention = LSHAttention::new(&config);

        let x = Array3::from_shape_fn((1, 256, 32), |_| rand_normal() * 0.1);
        let (output, _) = attention.forward(&x);

        assert_eq!(output.dim(), (1, 256, 32));

        // Check output is valid
        for &val in output.iter() {
            assert!(val.is_finite());
        }
    }
}
