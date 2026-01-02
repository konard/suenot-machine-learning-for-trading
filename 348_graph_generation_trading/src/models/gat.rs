//! Graph Attention Network implementation.
//!
//! GAT uses attention mechanisms to weight neighbor contributions.

use super::{Activation, FeatureMatrix, ModelOutput};
use crate::graph::MarketGraph;
use rand::Rng;

/// Graph Attention Layer
#[derive(Debug, Clone)]
pub struct GraphAttentionLayer {
    /// Weight matrix for linear transformation
    weights: Vec<Vec<f64>>,
    /// Attention weight vector (left)
    attention_left: Vec<f64>,
    /// Attention weight vector (right)
    attention_right: Vec<f64>,
    /// Number of attention heads
    num_heads: usize,
    /// Input dimension
    input_dim: usize,
    /// Output dimension per head
    output_dim: usize,
    /// Activation function
    activation: Activation,
    /// LeakyReLU negative slope for attention
    negative_slope: f64,
}

impl GraphAttentionLayer {
    /// Create a new GAT layer
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        num_heads: usize,
        activation: Activation,
    ) -> Self {
        let mut rng = rand::thread_rng();

        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();

        let weights: Vec<Vec<f64>> = (0..input_dim)
            .map(|_| {
                (0..output_dim * num_heads)
                    .map(|_| rng.gen_range(-scale..scale))
                    .collect()
            })
            .collect();

        let att_scale = (1.0 / output_dim as f64).sqrt();
        let attention_left: Vec<f64> = (0..output_dim * num_heads)
            .map(|_| rng.gen_range(-att_scale..att_scale))
            .collect();
        let attention_right: Vec<f64> = (0..output_dim * num_heads)
            .map(|_| rng.gen_range(-att_scale..att_scale))
            .collect();

        Self {
            weights,
            attention_left,
            attention_right,
            num_heads,
            input_dim,
            output_dim,
            activation,
            negative_slope: 0.2,
        }
    }

    /// Compute attention coefficients
    fn compute_attention(&self, h_i: &[f64], h_j: &[f64], head: usize) -> f64 {
        let start = head * self.output_dim;
        let end = start + self.output_dim;

        let mut score = 0.0;
        for k in start..end {
            score += self.attention_left[k] * h_i[k];
            score += self.attention_right[k] * h_j[k];
        }

        // LeakyReLU
        if score > 0.0 {
            score
        } else {
            self.negative_slope * score
        }
    }

    /// Forward pass through the layer
    pub fn forward(
        &self,
        features: &[Vec<f64>],
        adjacency: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        let n = features.len();
        let total_out_dim = self.output_dim * self.num_heads;

        // Linear transformation: H' = H @ W
        let mut transformed = vec![vec![0.0; total_out_dim]; n];
        for i in 0..n {
            for j in 0..total_out_dim {
                for k in 0..self.input_dim.min(features[i].len()) {
                    transformed[i][j] += features[i][k] * self.weights[k][j];
                }
            }
        }

        // Compute attention for each head
        let mut output = vec![vec![0.0; total_out_dim]; n];

        for head in 0..self.num_heads {
            // Compute attention coefficients
            let mut attention = vec![vec![0.0; n]; n];

            for i in 0..n {
                let mut neighbor_scores = Vec::new();
                let mut neighbor_indices = Vec::new();

                for j in 0..n {
                    if i == j || adjacency[i][j] > 0.0 {
                        let score = self.compute_attention(&transformed[i], &transformed[j], head);
                        neighbor_scores.push(score);
                        neighbor_indices.push(j);
                    }
                }

                // Softmax over neighbors
                if !neighbor_scores.is_empty() {
                    let max_score = neighbor_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exp_scores: Vec<f64> = neighbor_scores.iter().map(|s| (s - max_score).exp()).collect();
                    let sum_exp: f64 = exp_scores.iter().sum();

                    for (idx, &j) in neighbor_indices.iter().enumerate() {
                        attention[i][j] = exp_scores[idx] / sum_exp;
                    }
                }
            }

            // Aggregate with attention weights
            let start = head * self.output_dim;
            let end = start + self.output_dim;

            for i in 0..n {
                for j in 0..n {
                    if attention[i][j] > 0.0 {
                        for k in start..end {
                            output[i][k] += attention[i][j] * transformed[j][k];
                        }
                    }
                }
            }
        }

        // Apply activation
        for row in &mut output {
            for val in row {
                *val = self.activation.apply(*val);
            }
        }

        output
    }

    /// Get attention weights for visualization
    pub fn get_attention_weights(
        &self,
        features: &[Vec<f64>],
        adjacency: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        let n = features.len();
        let total_out_dim = self.output_dim * self.num_heads;

        // Transform features
        let mut transformed = vec![vec![0.0; total_out_dim]; n];
        for i in 0..n {
            for j in 0..total_out_dim {
                for k in 0..self.input_dim.min(features[i].len()) {
                    transformed[i][j] += features[i][k] * self.weights[k][j];
                }
            }
        }

        // Average attention across heads
        let mut avg_attention = vec![vec![0.0; n]; n];

        for head in 0..self.num_heads {
            for i in 0..n {
                let mut neighbor_scores = Vec::new();
                let mut neighbor_indices = Vec::new();

                for j in 0..n {
                    if i == j || adjacency[i][j] > 0.0 {
                        let score = self.compute_attention(&transformed[i], &transformed[j], head);
                        neighbor_scores.push(score);
                        neighbor_indices.push(j);
                    }
                }

                if !neighbor_scores.is_empty() {
                    let max_score = neighbor_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exp_scores: Vec<f64> = neighbor_scores.iter().map(|s| (s - max_score).exp()).collect();
                    let sum_exp: f64 = exp_scores.iter().sum();

                    for (idx, &j) in neighbor_indices.iter().enumerate() {
                        avg_attention[i][j] += exp_scores[idx] / sum_exp / self.num_heads as f64;
                    }
                }
            }
        }

        avg_attention
    }
}

/// Graph Attention Network
#[derive(Debug)]
pub struct GraphAttention {
    /// Layers
    layers: Vec<GraphAttentionLayer>,
    /// Number of heads
    num_heads: usize,
    /// Output dimension
    output_dim: usize,
}

impl GraphAttention {
    /// Create a new GAT
    pub fn new(
        input_dim: usize,
        hidden_dims: &[usize],
        output_dim: usize,
        num_heads: usize,
    ) -> Self {
        let mut layers = Vec::new();

        let mut prev_dim = input_dim;
        for &dim in hidden_dims {
            layers.push(GraphAttentionLayer::new(prev_dim, dim, num_heads, Activation::ReLU));
            prev_dim = dim * num_heads; // Output is concatenated heads
        }

        // Output layer with single head
        layers.push(GraphAttentionLayer::new(prev_dim, output_dim, 1, Activation::Linear));

        Self {
            layers,
            num_heads,
            output_dim,
        }
    }

    /// Forward pass
    pub fn forward(
        &self,
        graph: &MarketGraph,
        features: &FeatureMatrix,
    ) -> ModelOutput {
        let symbols = graph.symbols();
        let adjacency = graph.adjacency_matrix();

        let mut current = features.to_matrix(&symbols);

        for layer in &self.layers {
            current = layer.forward(&current, &adjacency);
        }

        let mut output = ModelOutput::new();
        for (i, symbol) in symbols.iter().enumerate() {
            let prediction = if current[i].is_empty() {
                0.0
            } else {
                current[i][0]
            };
            let confidence = prediction.abs().min(1.0);
            output.set_prediction(symbol, prediction, confidence);
        }

        output
    }

    /// Get attention weights from the first layer
    pub fn get_attention(
        &self,
        graph: &MarketGraph,
        features: &FeatureMatrix,
    ) -> Vec<Vec<f64>> {
        let symbols = graph.symbols();
        let adjacency = graph.adjacency_matrix();
        let feat_matrix = features.to_matrix(&symbols);

        if let Some(layer) = self.layers.first() {
            layer.get_attention_weights(&feat_matrix, &adjacency)
        } else {
            vec![]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> (MarketGraph, FeatureMatrix) {
        let mut graph = MarketGraph::new();
        graph.add_edge("BTC", "ETH", 0.8);
        graph.add_edge("ETH", "SOL", 0.7);

        let mut features = FeatureMatrix::new(3);
        features.set("BTC", vec![0.1, 0.2, 0.3]);
        features.set("ETH", vec![0.2, 0.3, 0.4]);
        features.set("SOL", vec![0.3, 0.4, 0.5]);

        (graph, features)
    }

    #[test]
    fn test_gat_layer() {
        let layer = GraphAttentionLayer::new(3, 2, 2, Activation::ReLU);

        let features = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.2, 0.3, 0.4],
        ];

        let adjacency = vec![
            vec![1.0, 0.8],
            vec![0.8, 1.0],
        ];

        let output = layer.forward(&features, &adjacency);
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 4); // 2 heads * 2 output dim
    }

    #[test]
    fn test_gat() {
        let (graph, features) = create_test_data();

        let gat = GraphAttention::new(3, &[4], 1, 2);
        let output = gat.forward(&graph, &features);

        assert!(output.predictions.contains_key("BTC"));
    }

    #[test]
    fn test_attention_weights() {
        let (graph, features) = create_test_data();

        let gat = GraphAttention::new(3, &[4], 1, 2);
        let attention = gat.get_attention(&graph, &features);

        // Attention weights should be non-negative and sum to ~1 per row
        for row in &attention {
            let sum: f64 = row.iter().sum();
            if sum > 0.0 {
                assert!((sum - 1.0).abs() < 0.1);
            }
        }
    }
}
