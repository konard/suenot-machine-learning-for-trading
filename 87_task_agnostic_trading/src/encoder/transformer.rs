//! Transformer encoder with self-attention for market data

use super::SharedEncoder;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde::{Deserialize, Serialize};

/// Transformer encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// Input feature dimension
    pub input_dim: usize,
    /// Output embedding dimension
    pub embedding_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Feed-forward dimension
    pub ff_dim: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Whether to use layer normalization
    pub use_layer_norm: bool,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            input_dim: 20,
            embedding_dim: 64,
            num_heads: 4,
            num_layers: 2,
            ff_dim: 128,
            dropout: 0.1,
            use_layer_norm: true,
        }
    }
}

/// Single attention head
struct AttentionHead {
    query_weights: Array2<f64>,
    key_weights: Array2<f64>,
    value_weights: Array2<f64>,
    head_dim: usize,
}

impl AttentionHead {
    fn new(input_dim: usize, head_dim: usize) -> Self {
        let scale = (2.0 / (input_dim + head_dim) as f64).sqrt();
        Self {
            query_weights: Array2::random((input_dim, head_dim), Uniform::new(-scale, scale)),
            key_weights: Array2::random((input_dim, head_dim), Uniform::new(-scale, scale)),
            value_weights: Array2::random((input_dim, head_dim), Uniform::new(-scale, scale)),
            head_dim,
        }
    }

    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        // Simple self-attention for single vector (treating it as a single token)
        let query = input.dot(&self.query_weights);
        let key = input.dot(&self.key_weights);
        let value = input.dot(&self.value_weights);

        // Attention score (scaled dot product)
        let score: f64 = query.iter().zip(key.iter()).map(|(q, k)| q * k).sum();
        let scale = (self.head_dim as f64).sqrt();
        let attention = (score / scale).exp();

        // Apply attention to value
        value.mapv(|v| v * attention / (attention + 1.0))
    }
}

/// Multi-head attention layer
struct MultiHeadAttention {
    heads: Vec<AttentionHead>,
    output_projection: Array2<f64>,
    embedding_dim: usize,
}

impl MultiHeadAttention {
    fn new(input_dim: usize, embedding_dim: usize, num_heads: usize) -> Self {
        let head_dim = embedding_dim / num_heads;
        let heads: Vec<_> = (0..num_heads)
            .map(|_| AttentionHead::new(input_dim, head_dim))
            .collect();

        let scale = (2.0 / (head_dim * num_heads + embedding_dim) as f64).sqrt();
        let output_projection = Array2::random(
            (head_dim * num_heads, embedding_dim),
            Uniform::new(-scale, scale),
        );

        Self {
            heads,
            output_projection,
            embedding_dim,
        }
    }

    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        // Concatenate outputs from all heads
        let head_outputs: Vec<f64> = self.heads
            .iter()
            .flat_map(|head| head.forward(input).to_vec())
            .collect();

        let concatenated = Array1::from_vec(head_outputs);
        concatenated.dot(&self.output_projection)
    }
}

/// Feed-forward network
struct FeedForward {
    w1: Array2<f64>,
    w2: Array2<f64>,
}

impl FeedForward {
    fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let scale1 = (2.0 / (input_dim + hidden_dim) as f64).sqrt();
        let scale2 = (2.0 / (hidden_dim + input_dim) as f64).sqrt();

        Self {
            w1: Array2::random((input_dim, hidden_dim), Uniform::new(-scale1, scale1)),
            w2: Array2::random((hidden_dim, input_dim), Uniform::new(-scale2, scale2)),
        }
    }

    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        // ReLU activation
        let hidden = input.dot(&self.w1).mapv(|x| x.max(0.0));
        hidden.dot(&self.w2)
    }
}

/// Transformer layer
struct TransformerLayer {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    use_layer_norm: bool,
    gamma1: Array1<f64>,
    beta1: Array1<f64>,
    gamma2: Array1<f64>,
    beta2: Array1<f64>,
}

impl TransformerLayer {
    fn new(dim: usize, num_heads: usize, ff_dim: usize, use_layer_norm: bool) -> Self {
        Self {
            attention: MultiHeadAttention::new(dim, dim, num_heads),
            feed_forward: FeedForward::new(dim, ff_dim),
            use_layer_norm,
            gamma1: Array1::ones(dim),
            beta1: Array1::zeros(dim),
            gamma2: Array1::ones(dim),
            beta2: Array1::zeros(dim),
        }
    }

    fn layer_norm(&self, x: &Array1<f64>, gamma: &Array1<f64>, beta: &Array1<f64>) -> Array1<f64> {
        let mean = x.mean().unwrap_or(0.0);
        let var = x.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(1.0);
        let std = (var + 1e-6).sqrt();
        x.mapv(|v| (v - mean) / std) * gamma + beta
    }

    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        // Self-attention with residual connection
        let attn_out = self.attention.forward(input);
        let x = input + &attn_out;
        let x = if self.use_layer_norm {
            self.layer_norm(&x, &self.gamma1, &self.beta1)
        } else {
            x
        };

        // Feed-forward with residual connection
        let ff_out = self.feed_forward.forward(&x);
        let x = &x + &ff_out;
        if self.use_layer_norm {
            self.layer_norm(&x, &self.gamma2, &self.beta2)
        } else {
            x
        }
    }
}

/// Transformer encoder for market data
pub struct TransformerEncoder {
    config: TransformerConfig,
    input_projection: Array2<f64>,
    layers: Vec<TransformerLayer>,
    output_projection: Array2<f64>,
}

impl TransformerEncoder {
    /// Create a new transformer encoder
    pub fn new(config: TransformerConfig) -> Self {
        let scale_in = (2.0 / (config.input_dim + config.embedding_dim) as f64).sqrt();
        let input_projection = Array2::random(
            (config.input_dim, config.embedding_dim),
            Uniform::new(-scale_in, scale_in),
        );

        let layers: Vec<_> = (0..config.num_layers)
            .map(|_| {
                TransformerLayer::new(
                    config.embedding_dim,
                    config.num_heads,
                    config.ff_dim,
                    config.use_layer_norm,
                )
            })
            .collect();

        let scale_out = (2.0 / (config.embedding_dim * 2) as f64).sqrt();
        let output_projection = Array2::random(
            (config.embedding_dim, config.embedding_dim),
            Uniform::new(-scale_out, scale_out),
        );

        Self {
            config,
            input_projection,
            layers,
            output_projection,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }
}

impl SharedEncoder for TransformerEncoder {
    fn encode(&self, input: &Array1<f64>) -> Array1<f64> {
        // Project input to embedding dimension
        let mut x = input.dot(&self.input_projection);

        // Pass through transformer layers
        for layer in &self.layers {
            x = layer.forward(&x);
        }

        // Final projection
        x.dot(&self.output_projection)
    }

    fn encode_batch(&self, inputs: &Array2<f64>) -> Array2<f64> {
        let mut outputs = Vec::with_capacity(inputs.nrows());
        for row in inputs.axis_iter(Axis(0)) {
            let embedding = self.encode(&row.to_owned());
            outputs.push(embedding);
        }

        // Stack outputs into 2D array
        let flat: Vec<f64> = outputs.iter().flat_map(|e| e.to_vec()).collect();
        Array2::from_shape_vec((inputs.nrows(), self.config.embedding_dim), flat)
            .expect("Shape mismatch in encode_batch")
    }

    fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    fn parameters(&self) -> Vec<Array2<f64>> {
        vec![
            self.input_projection.clone(),
            self.output_projection.clone(),
        ]
    }

    fn update_parameters(&mut self, gradients: &[Array2<f64>], learning_rate: f64) {
        if gradients.len() >= 2 {
            self.input_projection = &self.input_projection - &(&gradients[0] * learning_rate);
            self.output_projection = &self.output_projection - &(&gradients[1] * learning_rate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_transformer_encoder() {
        let config = TransformerConfig {
            input_dim: 10,
            embedding_dim: 32,
            num_heads: 4,
            num_layers: 2,
            ff_dim: 64,
            dropout: 0.1,
            use_layer_norm: true,
        };

        let encoder = TransformerEncoder::new(config);
        let input = Array::random(10, Uniform::new(-1.0, 1.0));
        let output = encoder.encode(&input);

        assert_eq!(output.len(), 32);
    }

    #[test]
    fn test_batch_encoding() {
        let config = TransformerConfig::default();
        let encoder = TransformerEncoder::new(config.clone());

        let batch = Array2::random((5, config.input_dim), Uniform::new(-1.0, 1.0));
        let outputs = encoder.encode_batch(&batch);

        assert_eq!(outputs.shape(), &[5, config.embedding_dim]);
    }
}
