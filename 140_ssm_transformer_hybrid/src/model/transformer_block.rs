//! Transformer block with multi-head self-attention and feed-forward network.
//!
//! Implements causal (decoder-style) attention where each position
//! can only attend to previous positions.

use rand_distr::{Distribution, Normal};

/// Multi-head self-attention layer.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    /// Query projection weights: [n_heads][head_dim][d_model]
    pub w_q: Vec<Vec<Vec<f64>>>,
    /// Key projection weights
    pub w_k: Vec<Vec<Vec<f64>>>,
    /// Value projection weights
    pub w_v: Vec<Vec<Vec<f64>>>,
    /// Output projection: [d_model][d_model]
    pub w_o: Vec<Vec<f64>>,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        let head_dim = d_model / n_heads;
        let mut rng = rand::thread_rng();
        let std = (1.0 / d_model as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let mut init_proj = |heads: usize, hd: usize, dm: usize| -> Vec<Vec<Vec<f64>>> {
            (0..heads)
                .map(|_| {
                    (0..hd)
                        .map(|_| (0..dm).map(|_| normal.sample(&mut rng)).collect())
                        .collect()
                })
                .collect()
        };

        let w_q = init_proj(n_heads, head_dim, d_model);
        let w_k = init_proj(n_heads, head_dim, d_model);
        let w_v = init_proj(n_heads, head_dim, d_model);

        let w_o = (0..d_model)
            .map(|_| (0..d_model).map(|_| normal.sample(&mut rng)).collect())
            .collect();

        Self {
            d_model,
            n_heads,
            head_dim,
            w_q,
            w_k,
            w_v,
            w_o,
        }
    }

    /// Compute self-attention with causal masking.
    ///
    /// # Arguments
    /// * `sequence` - Input sequence: Vec of timesteps, each Vec<f64> of d_model
    ///
    /// # Returns
    /// Attention output of same shape
    pub fn forward(&self, sequence: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let seq_len = sequence.len();
        let mut head_outputs: Vec<Vec<Vec<f64>>> = Vec::new();

        for head in 0..self.n_heads {
            // Project Q, K, V for this head
            let mut queries = Vec::with_capacity(seq_len);
            let mut keys = Vec::with_capacity(seq_len);
            let mut values = Vec::with_capacity(seq_len);

            for t in 0..seq_len {
                let x = &sequence[t];
                queries.push(self.project(x, &self.w_q[head]));
                keys.push(self.project(x, &self.w_k[head]));
                values.push(self.project(x, &self.w_v[head]));
            }

            // Compute causal attention
            let scale = (self.head_dim as f64).sqrt();
            let mut head_out = Vec::with_capacity(seq_len);

            for i in 0..seq_len {
                // Compute scores for position i attending to positions 0..=i
                let scores: Vec<f64> = (0..=i)
                    .map(|j| {
                        let dot: f64 = queries[i]
                            .iter()
                            .zip(keys[j].iter())
                            .map(|(q, k)| q * k)
                            .sum();
                        dot / scale
                    })
                    .collect();

                // Softmax
                let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_scores: Vec<f64> =
                    scores.iter().map(|&s| (s - max_score).exp()).collect();
                let sum_exp: f64 = exp_scores.iter().sum();
                let attn_weights: Vec<f64> =
                    exp_scores.iter().map(|&e| e / (sum_exp + 1e-10)).collect();

                // Weighted sum of values
                let mut out = vec![0.0; self.head_dim];
                for (j, weight) in attn_weights.iter().enumerate() {
                    for d in 0..self.head_dim {
                        out[d] += weight * values[j][d];
                    }
                }
                head_out.push(out);
            }
            head_outputs.push(head_out);
        }

        // Concatenate heads and project
        let mut output = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let mut concat = Vec::with_capacity(self.d_model);
            for head in 0..self.n_heads {
                concat.extend_from_slice(&head_outputs[head][t]);
            }
            // Output projection
            let projected: Vec<f64> = (0..self.d_model)
                .map(|o| {
                    self.w_o[o]
                        .iter()
                        .zip(concat.iter())
                        .map(|(w, x)| w * x)
                        .sum()
                })
                .collect();
            output.push(projected);
        }

        output
    }

    fn project(&self, x: &[f64], weights: &[Vec<f64>]) -> Vec<f64> {
        weights
            .iter()
            .map(|w| w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum())
            .collect()
    }

    pub fn num_parameters(&self) -> usize {
        let proj_params = self.n_heads * self.head_dim * self.d_model;
        proj_params * 3 + self.d_model * self.d_model // Q, K, V + output
    }
}

/// Feed-forward network with GELU activation.
#[derive(Debug, Clone)]
pub struct FeedForward {
    pub w1: Vec<Vec<f64>>,
    pub b1: Vec<f64>,
    pub w2: Vec<Vec<f64>>,
    pub b2: Vec<f64>,
    pub d_model: usize,
    pub d_ff: usize,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (d_model + d_ff) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let w1 = (0..d_ff)
            .map(|_| (0..d_model).map(|_| normal.sample(&mut rng)).collect())
            .collect();
        let b1 = vec![0.0; d_ff];

        let w2 = (0..d_model)
            .map(|_| (0..d_ff).map(|_| normal.sample(&mut rng)).collect())
            .collect();
        let b2 = vec![0.0; d_model];

        Self {
            w1, b1, w2, b2, d_model, d_ff,
        }
    }

    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        // First linear + GELU
        let hidden: Vec<f64> = (0..self.d_ff)
            .map(|i| {
                let dot: f64 = self.w1[i].iter().zip(x.iter()).map(|(w, xi)| w * xi).sum();
                gelu(dot + self.b1[i])
            })
            .collect();

        // Second linear
        (0..self.d_model)
            .map(|i| {
                let dot: f64 = self.w2[i]
                    .iter()
                    .zip(hidden.iter())
                    .map(|(w, h)| w * h)
                    .sum();
                dot + self.b2[i]
            })
            .collect()
    }

    pub fn num_parameters(&self) -> usize {
        self.d_model * self.d_ff + self.d_ff + self.d_ff * self.d_model + self.d_model
    }
}

/// GELU activation function.
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// Complete Transformer block: attention + FFN with residual connections.
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    pub attention: MultiHeadAttention,
    pub ffn: FeedForward,
    pub d_model: usize,
}

impl TransformerBlock {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        Self {
            attention: MultiHeadAttention::new(d_model, n_heads),
            ffn: FeedForward::new(d_model, d_ff),
            d_model,
        }
    }

    /// Process a sequence through attention + FFN with residual connections.
    pub fn forward(&self, sequence: &[Vec<f64>]) -> Vec<Vec<f64>> {
        // Self-attention with residual
        let attn_out = self.attention.forward(sequence);
        let after_attn: Vec<Vec<f64>> = sequence
            .iter()
            .zip(attn_out.iter())
            .map(|(x, a)| x.iter().zip(a.iter()).map(|(xi, ai)| xi + ai).collect())
            .collect();

        // FFN with residual
        after_attn
            .iter()
            .map(|x| {
                let ffn_out = self.ffn.forward(x);
                x.iter().zip(ffn_out.iter()).map(|(xi, fi)| xi + fi).collect()
            })
            .collect()
    }

    pub fn num_parameters(&self) -> usize {
        self.attention.num_parameters() + self.ffn.num_parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_head_attention() {
        let attn = MultiHeadAttention::new(8, 2);
        let seq: Vec<Vec<f64>> = (0..5).map(|_| vec![0.1; 8]).collect();
        let out = attn.forward(&seq);
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].len(), 8);
    }

    #[test]
    fn test_feed_forward() {
        let ffn = FeedForward::new(8, 32);
        let x = vec![0.1; 8];
        let out = ffn.forward(&x);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_transformer_block() {
        let block = TransformerBlock::new(8, 2, 32);
        let seq: Vec<Vec<f64>> = (0..5).map(|_| vec![0.1; 8]).collect();
        let out = block.forward(&seq);
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].len(), 8);
    }

    #[test]
    fn test_gelu() {
        assert!((gelu(0.0) - 0.0).abs() < 0.01);
        assert!(gelu(3.0) > 2.9);
    }
}
