//! Embedding module for TGN
//!
//! This module implements graph convolution and node embedding
//! generation for the temporal graph.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Embedding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Whether to use skip connections
    pub skip_connection: bool,
    /// Dropout rate
    pub dropout: f64,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            input_dim: 128,
            output_dim: 64,
            num_layers: 2,
            skip_connection: true,
            dropout: 0.1,
        }
    }
}

/// Graph convolution layer
#[derive(Debug)]
pub struct GraphConv {
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
    /// Weight matrix
    weights: Array2<f64>,
    /// Bias vector
    bias: Array1<f64>,
    /// Self-loop weight
    self_weight: Array2<f64>,
}

impl GraphConv {
    /// Create new graph convolution layer
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            weights: Array2::from_shape_fn((input_dim, output_dim), |_| {
                rand::random::<f64>() * 0.1 - 0.05
            }),
            bias: Array1::zeros(output_dim),
            self_weight: Array2::from_shape_fn((input_dim, output_dim), |_| {
                rand::random::<f64>() * 0.1 - 0.05
            }),
        }
    }

    /// Forward pass for a single node
    pub fn forward_single(&self, node_features: &Array1<f64>) -> Array1<f64> {
        // Handle dimension mismatch
        let features = if node_features.len() != self.input_dim {
            let mut padded = Array1::zeros(self.input_dim);
            let copy_len = node_features.len().min(self.input_dim);
            padded.slice_mut(ndarray::s![..copy_len])
                .assign(&node_features.slice(ndarray::s![..copy_len]));
            padded
        } else {
            node_features.clone()
        };

        // Apply self-loop transformation
        let output = features.dot(&self.self_weight) + &self.bias;

        // Apply ReLU
        output.mapv(|x| x.max(0.0))
    }

    /// Forward pass with neighbor aggregation
    pub fn forward_with_neighbors(
        &self,
        node_features: &Array1<f64>,
        neighbor_features: &[Array1<f64>],
        edge_weights: Option<&[f64]>,
    ) -> Array1<f64> {
        // Pad node features if needed
        let features = if node_features.len() != self.input_dim {
            let mut padded = Array1::zeros(self.input_dim);
            let copy_len = node_features.len().min(self.input_dim);
            padded.slice_mut(ndarray::s![..copy_len])
                .assign(&node_features.slice(ndarray::s![..copy_len]));
            padded
        } else {
            node_features.clone()
        };

        // Self transformation
        let self_contribution = features.dot(&self.self_weight);

        // Neighbor aggregation
        let neighbor_contribution = if neighbor_features.is_empty() {
            Array1::zeros(self.output_dim)
        } else {
            let weights = edge_weights.unwrap_or(&vec![1.0 / neighbor_features.len() as f64; neighbor_features.len()]);

            let mut aggregated = Array1::zeros(self.output_dim);
            for (neigh, &weight) in neighbor_features.iter().zip(weights.iter()) {
                // Pad neighbor features if needed
                let neigh_padded = if neigh.len() != self.input_dim {
                    let mut p = Array1::zeros(self.input_dim);
                    let copy_len = neigh.len().min(self.input_dim);
                    p.slice_mut(ndarray::s![..copy_len])
                        .assign(&neigh.slice(ndarray::s![..copy_len]));
                    p
                } else {
                    neigh.clone()
                };

                let transformed = neigh_padded.dot(&self.weights);
                aggregated = aggregated + transformed * weight;
            }
            aggregated
        };

        // Combine and apply non-linearity
        let output = self_contribution + neighbor_contribution + &self.bias;
        output.mapv(|x| x.max(0.0))
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Get number of parameters
    pub fn param_count(&self) -> usize {
        self.weights.len() + self.bias.len() + self.self_weight.len()
    }
}

/// Node embedding generator using stacked graph convolutions
#[derive(Debug)]
pub struct NodeEmbedding {
    /// Configuration
    config: EmbeddingConfig,
    /// Stacked layers
    layers: Vec<GraphConv>,
    /// Final projection layer
    output_layer: GraphConv,
}

impl NodeEmbedding {
    /// Create new node embedding module
    pub fn new(config: EmbeddingConfig) -> Self {
        let mut layers = Vec::new();
        let mut prev_dim = config.input_dim;

        // Create hidden layers
        for _ in 0..config.num_layers {
            let hidden_dim = (prev_dim + config.output_dim) / 2;
            layers.push(GraphConv::new(prev_dim, hidden_dim));
            prev_dim = hidden_dim;
        }

        // Output layer
        let output_layer = GraphConv::new(prev_dim, config.output_dim);

        Self {
            config,
            layers,
            output_layer,
        }
    }

    /// Generate embedding for a node
    pub fn embed(
        &self,
        node_features: &Array1<f64>,
        neighbor_features: &[Array1<f64>],
    ) -> Array1<f64> {
        let mut h = node_features.clone();

        // Apply stacked layers
        for layer in &self.layers {
            let new_h = layer.forward_with_neighbors(&h, neighbor_features, None);

            // Skip connection if enabled
            if self.config.skip_connection && h.len() == new_h.len() {
                h = &h + &new_h;
            } else {
                h = new_h;
            }
        }

        // Final projection
        self.output_layer.forward_single(&h)
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.config.output_dim
    }

    /// Get total parameter count
    pub fn param_count(&self) -> usize {
        let layer_params: usize = self.layers.iter().map(|l| l.param_count()).sum();
        layer_params + self.output_layer.param_count()
    }
}

/// Graph Attention layer for neighbor aggregation
#[derive(Debug)]
pub struct GraphAttentionLayer {
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Weight matrices per head
    head_weights: Vec<Array2<f64>>,
    /// Attention weight vectors per head
    attention_weights: Vec<Array1<f64>>,
}

impl GraphAttentionLayer {
    /// Create new graph attention layer
    pub fn new(input_dim: usize, output_dim: usize, num_heads: usize) -> Self {
        let head_dim = output_dim / num_heads;

        let head_weights: Vec<Array2<f64>> = (0..num_heads)
            .map(|_| {
                Array2::from_shape_fn((input_dim, head_dim), |_| {
                    rand::random::<f64>() * 0.1 - 0.05
                })
            })
            .collect();

        let attention_weights: Vec<Array1<f64>> = (0..num_heads)
            .map(|_| {
                Array1::from_shape_fn(head_dim * 2, |_| {
                    rand::random::<f64>() * 0.1 - 0.05
                })
            })
            .collect();

        Self {
            input_dim,
            output_dim,
            num_heads,
            head_weights,
            attention_weights,
        }
    }

    /// Compute attention-weighted aggregation
    pub fn forward(
        &self,
        node_features: &Array1<f64>,
        neighbor_features: &[Array1<f64>],
    ) -> Array1<f64> {
        if neighbor_features.is_empty() {
            // Just transform the node features
            let mut output = Array1::zeros(self.output_dim);
            let head_dim = self.output_dim / self.num_heads;

            for (h, weights) in self.head_weights.iter().enumerate() {
                // Pad if needed
                let features = if node_features.len() != self.input_dim {
                    let mut p = Array1::zeros(self.input_dim);
                    let copy_len = node_features.len().min(self.input_dim);
                    p.slice_mut(ndarray::s![..copy_len])
                        .assign(&node_features.slice(ndarray::s![..copy_len]));
                    p
                } else {
                    node_features.clone()
                };

                let head_output = features.dot(weights);
                output.slice_mut(ndarray::s![h * head_dim..(h + 1) * head_dim])
                    .assign(&head_output);
            }

            return output.mapv(|x| x.max(0.0));
        }

        let mut output = Array1::zeros(self.output_dim);
        let head_dim = self.output_dim / self.num_heads;

        for (h, (weights, attn_weights)) in self.head_weights.iter()
            .zip(self.attention_weights.iter())
            .enumerate()
        {
            // Pad node features if needed
            let node_padded = if node_features.len() != self.input_dim {
                let mut p = Array1::zeros(self.input_dim);
                let copy_len = node_features.len().min(self.input_dim);
                p.slice_mut(ndarray::s![..copy_len])
                    .assign(&node_features.slice(ndarray::s![..copy_len]));
                p
            } else {
                node_features.clone()
            };

            // Transform node
            let h_i = node_padded.dot(weights);

            // Compute attention scores for each neighbor
            let mut scores: Vec<f64> = Vec::with_capacity(neighbor_features.len());
            let mut transformed_neighbors: Vec<Array1<f64>> = Vec::with_capacity(neighbor_features.len());

            for neigh in neighbor_features {
                // Pad neighbor features if needed
                let neigh_padded = if neigh.len() != self.input_dim {
                    let mut p = Array1::zeros(self.input_dim);
                    let copy_len = neigh.len().min(self.input_dim);
                    p.slice_mut(ndarray::s![..copy_len])
                        .assign(&neigh.slice(ndarray::s![..copy_len]));
                    p
                } else {
                    neigh.clone()
                };

                let h_j = neigh_padded.dot(weights);

                // Concatenate and compute attention score
                let mut concat = Array1::zeros(head_dim * 2);
                concat.slice_mut(ndarray::s![..head_dim]).assign(&h_i);
                concat.slice_mut(ndarray::s![head_dim..]).assign(&h_j);

                let score = concat.dot(attn_weights);
                let score = leaky_relu(score, 0.2);

                scores.push(score);
                transformed_neighbors.push(h_j);
            }

            // Softmax attention weights
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exp_scores.iter().sum();
            let attention: Vec<f64> = exp_scores.iter().map(|&e| e / sum_exp).collect();

            // Weighted aggregation
            let mut head_output = Array1::zeros(head_dim);
            for (neigh, &attn) in transformed_neighbors.iter().zip(attention.iter()) {
                head_output = head_output + neigh * attn;
            }

            // Add self contribution
            head_output = head_output + &h_i;

            // Store in output
            output.slice_mut(ndarray::s![h * head_dim..(h + 1) * head_dim])
                .assign(&head_output);
        }

        // Apply activation
        output.mapv(|x| x.max(0.0))
    }

    /// Get number of parameters
    pub fn param_count(&self) -> usize {
        let weight_params: usize = self.head_weights.iter().map(|w| w.len()).sum();
        let attn_params: usize = self.attention_weights.iter().map(|a| a.len()).sum();
        weight_params + attn_params
    }
}

/// Leaky ReLU activation
fn leaky_relu(x: f64, alpha: f64) -> f64 {
    if x > 0.0 { x } else { alpha * x }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_conv() {
        let layer = GraphConv::new(16, 8);
        let features = Array1::from_shape_fn(16, |_| rand::random::<f64>());

        let output = layer.forward_single(&features);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_graph_conv_with_neighbors() {
        let layer = GraphConv::new(16, 8);
        let node = Array1::from_shape_fn(16, |_| rand::random::<f64>());
        let neighbors: Vec<Array1<f64>> = (0..3)
            .map(|_| Array1::from_shape_fn(16, |_| rand::random::<f64>()))
            .collect();

        let output = layer.forward_with_neighbors(&node, &neighbors, None);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_node_embedding() {
        let config = EmbeddingConfig {
            input_dim: 16,
            output_dim: 8,
            num_layers: 2,
            skip_connection: true,
            dropout: 0.0,
        };

        let embedding = NodeEmbedding::new(config);
        let node = Array1::from_shape_fn(16, |_| rand::random::<f64>());
        let neighbors: Vec<Array1<f64>> = (0..3)
            .map(|_| Array1::from_shape_fn(16, |_| rand::random::<f64>()))
            .collect();

        let output = embedding.embed(&node, &neighbors);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_graph_attention() {
        let layer = GraphAttentionLayer::new(16, 8, 2);
        let node = Array1::from_shape_fn(16, |_| rand::random::<f64>());
        let neighbors: Vec<Array1<f64>> = (0..3)
            .map(|_| Array1::from_shape_fn(16, |_| rand::random::<f64>()))
            .collect();

        let output = layer.forward(&node, &neighbors);
        assert_eq!(output.len(), 8);
    }
}
