//! Transformer model implementation with attention weight extraction.
//!
//! Implements a Transformer encoder that returns attention weights alongside
//! predictions, enabling visualization and interpretation of model decisions.

use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

/// Configuration for the Transformer model.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// Number of input features
    pub input_dim: usize,
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of transformer layers
    pub n_layers: usize,
    /// Feed-forward dimension
    pub d_ff: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Maximum sequence length
    pub max_seq_len: usize,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            input_dim: 8,
            d_model: 64,
            n_heads: 4,
            n_layers: 3,
            d_ff: 256,
            dropout: 0.1,
            max_seq_len: 512,
        }
    }
}

/// Container for attention weights from all layers.
#[derive(Debug, Clone)]
pub struct AttentionWeights {
    /// Attention weights per layer: Vec<(batch, heads, seq_len, seq_len)>
    pub layers: Vec<Vec<Vec<Vec<Vec<f64>>>>>,
}

impl AttentionWeights {
    /// Create empty attention weights.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Compute entropy of attention distribution.
    pub fn compute_entropy(&self) -> f64 {
        if self.layers.is_empty() {
            return 0.0;
        }

        let mut total_entropy = 0.0;
        let mut count = 0;

        for layer in &self.layers {
            for batch in layer {
                for head in batch {
                    for row in head {
                        let entropy: f64 = row
                            .iter()
                            .filter(|&&w| w > 1e-9)
                            .map(|&w| -w * w.ln())
                            .sum();
                        total_entropy += entropy;
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            total_entropy / count as f64
        } else {
            0.0
        }
    }

    /// Compute confidence score based on attention focus.
    pub fn compute_confidence(&self, seq_len: usize) -> f64 {
        let entropy = self.compute_entropy();
        let max_entropy = (seq_len as f64).ln();
        1.0 - (entropy / max_entropy).min(1.0)
    }

    /// Get average attention from last layer.
    pub fn get_last_layer_attention(&self) -> Option<Vec<Vec<f64>>> {
        self.layers.last().map(|layer| {
            // Average over batch and heads
            let batch_size = layer.len();
            let n_heads = layer.get(0).map(|b| b.len()).unwrap_or(0);
            let seq_len = layer.get(0).and_then(|b| b.get(0)).map(|h| h.len()).unwrap_or(0);

            if batch_size == 0 || n_heads == 0 || seq_len == 0 {
                return vec![vec![0.0; seq_len]; seq_len];
            }

            let mut avg = vec![vec![0.0; seq_len]; seq_len];
            for batch in layer {
                for head in batch {
                    for (i, row) in head.iter().enumerate() {
                        for (j, &val) in row.iter().enumerate() {
                            avg[i][j] += val;
                        }
                    }
                }
            }

            let divisor = (batch_size * n_heads) as f64;
            for row in &mut avg {
                for val in row {
                    *val /= divisor;
                }
            }

            avg
        })
    }
}

impl Default for AttentionWeights {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-head attention mechanism.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    d_model: usize,
    n_heads: usize,
    d_k: usize,
    // Weight matrices (simplified for demonstration)
    w_q: Vec<Vec<f64>>,
    w_k: Vec<Vec<f64>>,
    w_v: Vec<Vec<f64>>,
    w_o: Vec<Vec<f64>>,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention module.
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        let d_k = d_model / n_heads;
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        // Initialize weights with small random values
        let init_matrix = |rows: usize, cols: usize| -> Vec<Vec<f64>> {
            (0..rows)
                .map(|_| (0..cols).map(|_| normal.sample(&mut rng)).collect())
                .collect()
        };

        Self {
            d_model,
            n_heads,
            d_k,
            w_q: init_matrix(d_model, d_model),
            w_k: init_matrix(d_model, d_model),
            w_v: init_matrix(d_model, d_model),
            w_o: init_matrix(d_model, d_model),
        }
    }

    /// Forward pass through the attention mechanism.
    pub fn forward(&self, x: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>) {
        let seq_len = x.len();

        // Project Q, K, V
        let q = self.linear_transform(x, &self.w_q);
        let k = self.linear_transform(x, &self.w_k);
        let v = self.linear_transform(x, &self.w_v);

        // Split into heads and compute attention
        let mut all_head_outputs = Vec::new();
        let mut all_head_attention = Vec::new();
        let scale = (self.d_k as f64).sqrt();

        for h in 0..self.n_heads {
            let start = h * self.d_k;
            let end = start + self.d_k;

            // Extract head projections
            let q_h: Vec<Vec<f64>> = q.iter().map(|row| row[start..end].to_vec()).collect();
            let k_h: Vec<Vec<f64>> = k.iter().map(|row| row[start..end].to_vec()).collect();
            let v_h: Vec<Vec<f64>> = v.iter().map(|row| row[start..end].to_vec()).collect();

            // Compute attention scores
            let mut scores = vec![vec![0.0; seq_len]; seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let dot: f64 = q_h[i].iter().zip(&k_h[j]).map(|(a, b)| a * b).sum();
                    scores[i][j] = dot / scale;
                }
            }

            // Softmax
            let attention = self.softmax_2d(&scores);
            all_head_attention.push(attention.clone());

            // Apply attention to values
            let mut head_output = vec![vec![0.0; self.d_k]; seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    for k in 0..self.d_k {
                        head_output[i][k] += attention[i][j] * v_h[j][k];
                    }
                }
            }
            all_head_outputs.push(head_output);
        }

        // Concatenate heads
        let mut concat = vec![vec![0.0; self.d_model]; seq_len];
        for (h, head_output) in all_head_outputs.iter().enumerate() {
            for i in 0..seq_len {
                for k in 0..self.d_k {
                    concat[i][h * self.d_k + k] = head_output[i][k];
                }
            }
        }

        // Output projection
        let output = self.linear_transform(&concat, &self.w_o);

        (output, all_head_attention)
    }

    fn linear_transform(&self, x: &[Vec<f64>], w: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let seq_len = x.len();
        let out_dim = w[0].len();
        let mut result = vec![vec![0.0; out_dim]; seq_len];

        for i in 0..seq_len {
            for j in 0..out_dim {
                for k in 0..x[0].len().min(w.len()) {
                    result[i][j] += x[i][k] * w[k][j];
                }
            }
        }

        result
    }

    fn softmax_2d(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter()
            .map(|row| {
                let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp: Vec<f64> = row.iter().map(|&v| (v - max).exp()).collect();
                let sum: f64 = exp.iter().sum();
                exp.iter().map(|&v| v / sum).collect()
            })
            .collect()
    }
}

/// Transformer encoder with attention extraction.
#[derive(Debug)]
pub struct AttentionTransformer {
    config: TransformerConfig,
    input_proj: Vec<Vec<f64>>,
    layers: Vec<MultiHeadAttention>,
    output_head: Vec<Vec<f64>>,
    pos_encoding: Vec<Vec<f64>>,
}

impl AttentionTransformer {
    /// Create a new Transformer model.
    pub fn new(config: TransformerConfig) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        // Input projection
        let input_proj: Vec<Vec<f64>> = (0..config.input_dim)
            .map(|_| (0..config.d_model).map(|_| normal.sample(&mut rng)).collect())
            .collect();

        // Transformer layers
        let layers: Vec<MultiHeadAttention> = (0..config.n_layers)
            .map(|_| MultiHeadAttention::new(config.d_model, config.n_heads))
            .collect();

        // Output head (for 3 outputs: return, direction, volatility)
        let output_head: Vec<Vec<f64>> = (0..config.d_model)
            .map(|_| (0..3).map(|_| normal.sample(&mut rng)).collect())
            .collect();

        // Positional encoding
        let pos_encoding = Self::create_positional_encoding(config.max_seq_len, config.d_model);

        Self {
            config,
            input_proj,
            layers,
            output_head,
            pos_encoding,
        }
    }

    /// Create sinusoidal positional encoding.
    fn create_positional_encoding(max_len: usize, d_model: usize) -> Vec<Vec<f64>> {
        let mut pe = vec![vec![0.0; d_model]; max_len];

        for pos in 0..max_len {
            for i in 0..d_model {
                let angle = pos as f64 / (10000_f64.powf((2 * (i / 2)) as f64 / d_model as f64));
                pe[pos][i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }

        pe
    }

    /// Forward pass through the transformer.
    pub fn forward(&self, x: &[Vec<f64>]) -> (f64, f64, f64, AttentionWeights) {
        let seq_len = x.len();

        // Input projection
        let mut hidden = self.project_input(x);

        // Add positional encoding
        for i in 0..seq_len.min(self.pos_encoding.len()) {
            for j in 0..self.config.d_model {
                hidden[i][j] += self.pos_encoding[i][j];
            }
        }

        // Process through layers
        let mut attention_weights = AttentionWeights::new();

        for layer in &self.layers {
            let (output, attn) = layer.forward(&hidden);
            // Wrap attention in batch dimension
            attention_weights.layers.push(vec![attn]);

            // Residual connection (simplified)
            for i in 0..seq_len {
                for j in 0..self.config.d_model {
                    hidden[i][j] = hidden[i][j] * 0.5 + output[i][j] * 0.5;
                }
            }
        }

        // Use last hidden state for prediction
        let last_hidden = &hidden[seq_len - 1];

        // Output projection
        let mut outputs = vec![0.0; 3];
        for (j, out) in outputs.iter_mut().enumerate() {
            for (i, &h) in last_hidden.iter().enumerate() {
                if i < self.output_head.len() {
                    *out += h * self.output_head[i][j];
                }
            }
        }

        // Apply activations
        let pred_return = outputs[0]; // Raw return prediction
        let pred_direction = 1.0 / (1.0 + (-outputs[1]).exp()); // Sigmoid
        let pred_volatility = outputs[2].exp(); // Softplus approximation

        (pred_return, pred_direction, pred_volatility, attention_weights)
    }

    fn project_input(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let seq_len = x.len();
        let mut result = vec![vec![0.0; self.config.d_model]; seq_len];

        for i in 0..seq_len {
            for j in 0..self.config.d_model {
                for k in 0..x[0].len().min(self.input_proj.len()) {
                    result[i][j] += x[i][k] * self.input_proj[k][j];
                }
            }
        }

        result
    }

    /// Get model configuration.
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_forward() {
        let config = TransformerConfig {
            input_dim: 8,
            d_model: 32,
            n_heads: 4,
            n_layers: 2,
            ..Default::default()
        };

        let model = AttentionTransformer::new(config);

        // Create dummy input (seq_len=10, features=8)
        let input: Vec<Vec<f64>> = (0..10)
            .map(|_| (0..8).map(|_| rand::random::<f64>()).collect())
            .collect();

        let (ret, dir, vol, attn) = model.forward(&input);

        assert!(dir >= 0.0 && dir <= 1.0, "Direction should be in [0, 1]");
        assert!(vol >= 0.0, "Volatility should be non-negative");
        assert_eq!(attn.layers.len(), 2, "Should have attention from 2 layers");
    }

    #[test]
    fn test_attention_confidence() {
        let mut weights = AttentionWeights::new();

        // Create focused attention (one element = 1.0, rest = 0)
        let focused: Vec<Vec<Vec<Vec<f64>>>> = vec![vec![vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ]]];
        weights.layers = focused;

        let confidence = weights.compute_confidence(4);
        assert!(confidence > 0.9, "Focused attention should have high confidence");
    }
}
