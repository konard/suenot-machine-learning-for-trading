//! BigBird Sparse Attention Implementation
//!
//! This module implements the sparse attention mechanism from the BigBird paper.
//! It combines three types of attention:
//! 1. Random attention - connections to randomly selected tokens
//! 2. Window attention - local sliding window attention
//! 3. Global attention - tokens that attend to/from all positions

use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    prelude::*,
    tensor::activation::softmax,
};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

/// BigBird Sparse Attention module
#[derive(Module, Debug)]
pub struct BigBirdSparseAttention<B: Backend> {
    /// Query projection
    query: Linear<B>,
    /// Key projection
    key: Linear<B>,
    /// Value projection
    value: Linear<B>,
    /// Output projection
    output: Linear<B>,
    /// Dropout layer
    dropout: Dropout,
    /// Number of attention heads
    n_heads: usize,
    /// Dimension per head
    head_dim: usize,
    /// Window size for local attention
    window_size: usize,
    /// Number of random connections
    num_random: usize,
    /// Number of global tokens
    num_global: usize,
    /// Sequence length
    seq_len: usize,
    /// Random seed for reproducible random patterns
    seed: u64,
}

impl<B: Backend> BigBirdSparseAttention<B> {
    /// Create a new BigBird sparse attention module
    pub fn new(
        device: &B::Device,
        d_model: usize,
        n_heads: usize,
        seq_len: usize,
        window_size: usize,
        num_random: usize,
        num_global: usize,
        dropout: f32,
        seed: u64,
    ) -> Self {
        let head_dim = d_model / n_heads;

        let query = LinearConfig::new(d_model, d_model).init(device);
        let key = LinearConfig::new(d_model, d_model).init(device);
        let value = LinearConfig::new(d_model, d_model).init(device);
        let output = LinearConfig::new(d_model, d_model).init(device);
        let dropout = DropoutConfig::new(dropout as f64).init();

        Self {
            query,
            key,
            value,
            output,
            dropout,
            n_heads,
            head_dim,
            window_size,
            num_random,
            num_global,
            seq_len,
            seed,
        }
    }

    /// Generate the sparse attention mask
    ///
    /// The mask combines:
    /// 1. Window (local) attention - each token attends to nearby tokens
    /// 2. Random attention - each token attends to random tokens
    /// 3. Global attention - first `num_global` tokens attend to/from all positions
    fn create_attention_mask(&self, device: &B::Device) -> Tensor<B, 2> {
        let seq_len = self.seq_len;
        let mut mask = vec![vec![0.0f32; seq_len]; seq_len];

        // 1. Window (local) attention
        let half_window = self.window_size / 2;
        for i in 0..seq_len {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(seq_len);
            for j in start..end {
                mask[i][j] = 1.0;
            }
        }

        // 2. Random attention (using seeded RNG for reproducibility)
        let mut rng = StdRng::seed_from_u64(self.seed);
        for i in 0..seq_len {
            // Skip global tokens
            if i < self.num_global {
                continue;
            }

            // Get indices not already connected
            let mut available: Vec<usize> = (0..seq_len)
                .filter(|&j| mask[i][j] == 0.0 && j >= self.num_global)
                .collect();

            // Randomly select num_random connections
            available.shuffle(&mut rng);
            for &j in available.iter().take(self.num_random) {
                mask[i][j] = 1.0;
                mask[j][i] = 1.0; // Make symmetric
            }
        }

        // 3. Global attention - first num_global tokens
        for i in 0..self.num_global {
            for j in 0..seq_len {
                mask[i][j] = 1.0; // Global token attends to all
                mask[j][i] = 1.0; // All attend to global token
            }
        }

        // Flatten and create tensor
        let flat_mask: Vec<f32> = mask.into_iter().flatten().collect();
        Tensor::from_floats(flat_mask.as_slice(), device).reshape([seq_len, seq_len])
    }

    /// Forward pass through the sparse attention layer
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = x.device();
        let [batch_size, seq_len, _d_model] = x.dims();

        // Project to Q, K, V
        let q = self.query.forward(x.clone());
        let k = self.key.forward(x.clone());
        let v = self.value.forward(x);

        // Reshape for multi-head attention: [batch, seq, d_model] -> [batch, heads, seq, head_dim]
        let q = q
            .reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);

        // Compute attention scores: Q @ K^T / sqrt(head_dim)
        let scale = (self.head_dim as f32).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)) / scale;

        // Apply sparse attention mask
        let mask = self.create_attention_mask(&device);
        let mask = mask
            .unsqueeze::<3>()
            .unsqueeze::<4>()
            .expand([batch_size, self.n_heads, seq_len, seq_len]);

        // Mask out invalid positions (set to -inf before softmax)
        let neg_inf = Tensor::full([batch_size, self.n_heads, seq_len, seq_len], f32::NEG_INFINITY, &device);
        let ones = Tensor::ones([batch_size, self.n_heads, seq_len, seq_len], &device);
        let scores = scores.clone() * mask.clone() + neg_inf * (ones - mask);

        // Softmax and dropout
        let attn_weights = softmax(scores, 3);
        let attn_weights = self.dropout.forward(attn_weights);

        // Apply attention to values
        let output = attn_weights.matmul(v);

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, d_model]
        let output = output
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, self.n_heads * self.head_dim]);

        // Output projection
        self.output.forward(output)
    }

    /// Get attention statistics for analysis
    pub fn get_attention_stats(&self) -> AttentionStats {
        let total_positions = self.seq_len * self.seq_len;
        let window_connections = self.seq_len * self.window_size;
        let random_connections = (self.seq_len - self.num_global) * self.num_random * 2;
        let global_connections = self.num_global * self.seq_len * 2;

        // Approximate number of non-zero positions (with some overlap)
        let sparse_connections = (window_connections + random_connections + global_connections)
            .min(total_positions);

        AttentionStats {
            seq_len: self.seq_len,
            window_size: self.window_size,
            num_random: self.num_random,
            num_global: self.num_global,
            total_positions,
            sparse_connections,
            sparsity: 1.0 - (sparse_connections as f32 / total_positions as f32),
            memory_savings: sparse_connections as f32 / total_positions as f32,
        }
    }
}

/// Statistics about the attention pattern
#[derive(Debug, Clone)]
pub struct AttentionStats {
    pub seq_len: usize,
    pub window_size: usize,
    pub num_random: usize,
    pub num_global: usize,
    pub total_positions: usize,
    pub sparse_connections: usize,
    pub sparsity: f32,
    pub memory_savings: f32,
}

impl std::fmt::Display for AttentionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AttentionStats {{ seq_len: {}, window: {}, random: {}, global: {}, sparsity: {:.2}%, memory_savings: {:.2}x }}",
            self.seq_len,
            self.window_size,
            self.num_random,
            self.num_global,
            self.sparsity * 100.0,
            1.0 / self.memory_savings
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_attention_creation() {
        let device = Default::default();
        let attention = BigBirdSparseAttention::<TestBackend>::new(
            &device,
            64,   // d_model
            4,    // n_heads
            128,  // seq_len
            7,    // window_size
            3,    // num_random
            2,    // num_global
            0.1,  // dropout
            42,   // seed
        );

        let stats = attention.get_attention_stats();
        assert!(stats.sparsity > 0.5); // Should be sparse
        println!("{}", stats);
    }

    #[test]
    fn test_forward_pass() {
        let device = Default::default();
        let attention = BigBirdSparseAttention::<TestBackend>::new(
            &device,
            64,   // d_model
            4,    // n_heads
            32,   // seq_len (smaller for test)
            7,    // window_size
            3,    // num_random
            2,    // num_global
            0.0,  // dropout (0 for deterministic test)
            42,   // seed
        );

        let x = Tensor::random([2, 32, 64], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let output = attention.forward(x.clone());

        assert_eq!(output.dims(), [2, 32, 64]);
    }
}
