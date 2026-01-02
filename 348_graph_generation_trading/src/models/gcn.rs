//! Graph Convolutional Network implementation.
//!
//! Simplified GCN for trading signal generation.

use super::{Activation, FeatureMatrix, ModelOutput};
use crate::graph::MarketGraph;
use rand::Rng;

/// Graph Convolutional Layer
#[derive(Debug, Clone)]
pub struct GraphConvLayer {
    /// Weight matrix
    weights: Vec<Vec<f64>>,
    /// Bias vector
    bias: Vec<f64>,
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
    /// Activation function
    activation: Activation,
}

impl GraphConvLayer {
    /// Create a new layer with random initialization
    pub fn new(input_dim: usize, output_dim: usize, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();

        let weights: Vec<Vec<f64>> = (0..input_dim)
            .map(|_| {
                (0..output_dim)
                    .map(|_| rng.gen_range(-scale..scale))
                    .collect()
            })
            .collect();

        let bias = vec![0.0; output_dim];

        Self {
            weights,
            bias,
            input_dim,
            output_dim,
            activation,
        }
    }

    /// Forward pass through the layer
    ///
    /// H' = σ(D^(-1/2) A D^(-1/2) H W + b)
    pub fn forward(
        &self,
        features: &[Vec<f64>],
        adjacency: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        let n = features.len();

        // Calculate degree matrix
        let degrees: Vec<f64> = adjacency
            .iter()
            .map(|row| row.iter().sum::<f64>() + 1.0) // +1 for self-loop
            .collect();

        // Normalized adjacency (with self-loops)
        let mut norm_adj = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let a = if i == j {
                    adjacency[i][j] + 1.0 // Self-loop
                } else {
                    adjacency[i][j]
                };
                norm_adj[i][j] = a / (degrees[i] * degrees[j]).sqrt();
            }
        }

        // Aggregate neighbor features: H_agg = norm_adj @ features
        let mut aggregated = vec![vec![0.0; self.input_dim]; n];
        for i in 0..n {
            for j in 0..n {
                if norm_adj[i][j] > 0.0 {
                    for k in 0..self.input_dim.min(features[j].len()) {
                        aggregated[i][k] += norm_adj[i][j] * features[j][k];
                    }
                }
            }
        }

        // Apply weights and activation: H' = σ(H_agg @ W + b)
        let mut output = vec![vec![0.0; self.output_dim]; n];
        for i in 0..n {
            for j in 0..self.output_dim {
                let mut sum = self.bias[j];
                for k in 0..self.input_dim.min(aggregated[i].len()) {
                    sum += aggregated[i][k] * self.weights[k][j];
                }
                output[i][j] = self.activation.apply(sum);
            }
        }

        output
    }
}

/// Graph Convolutional Network
#[derive(Debug)]
pub struct GraphConvolution {
    /// Layers
    layers: Vec<GraphConvLayer>,
    /// Output dimension
    output_dim: usize,
}

impl GraphConvolution {
    /// Create a new GCN with specified architecture
    pub fn new(input_dim: usize, hidden_dims: &[usize], output_dim: usize) -> Self {
        let mut layers = Vec::new();

        let mut prev_dim = input_dim;
        for &dim in hidden_dims {
            layers.push(GraphConvLayer::new(prev_dim, dim, Activation::ReLU));
            prev_dim = dim;
        }

        // Output layer with linear activation
        layers.push(GraphConvLayer::new(prev_dim, output_dim, Activation::Linear));

        Self { layers, output_dim }
    }

    /// Forward pass through the network
    pub fn forward(
        &self,
        graph: &MarketGraph,
        features: &FeatureMatrix,
    ) -> ModelOutput {
        let symbols = graph.symbols();
        let adjacency = graph.adjacency_matrix();

        // Get feature matrix
        let mut current_features = features.to_matrix(&symbols);

        // Forward through layers
        for layer in &self.layers {
            current_features = layer.forward(&current_features, &adjacency);
        }

        // Convert to output
        let mut output = ModelOutput::new();
        for (i, symbol) in symbols.iter().enumerate() {
            let prediction = if current_features[i].is_empty() {
                0.0
            } else {
                current_features[i][0]
            };

            // Confidence based on prediction magnitude
            let confidence = prediction.abs().min(1.0);
            output.set_prediction(symbol, prediction, confidence);
        }

        output
    }

    /// Predict direction (long/short signal)
    pub fn predict_direction(
        &self,
        graph: &MarketGraph,
        features: &FeatureMatrix,
    ) -> ModelOutput {
        let raw_output = self.forward(graph, features);

        let mut output = ModelOutput::new();
        for (symbol, &pred) in &raw_output.predictions {
            let direction = if pred > 0.0 { 1.0 } else { -1.0 };
            let confidence = raw_output.confidence.get(symbol).copied().unwrap_or(0.5);
            output.set_prediction(symbol, direction, confidence);
        }

        output
    }
}

/// Simple GCN-based feature extractor
pub fn gcn_features(
    graph: &MarketGraph,
    features: &FeatureMatrix,
    num_layers: usize,
) -> FeatureMatrix {
    let symbols = graph.symbols();
    let adjacency = graph.adjacency_matrix();
    let input_dim = features.feature_dim();

    let mut current = features.to_matrix(&symbols);

    // Simple message passing without learned weights
    for _ in 0..num_layers {
        let n = current.len();
        let dim = if current.is_empty() { 0 } else { current[0].len() };

        // Calculate degrees
        let degrees: Vec<f64> = adjacency
            .iter()
            .map(|row| row.iter().sum::<f64>() + 1.0)
            .collect();

        // Aggregate
        let mut new_features = vec![vec![0.0; dim]; n];
        for i in 0..n {
            for j in 0..n {
                let a = if i == j {
                    1.0
                } else if adjacency[i][j] > 0.0 {
                    adjacency[i][j]
                } else {
                    0.0
                };

                if a > 0.0 {
                    let norm = (degrees[i] * degrees[j]).sqrt();
                    for k in 0..dim {
                        new_features[i][k] += (a / norm) * current[j][k];
                    }
                }
            }
        }

        current = new_features;
    }

    // Convert back to feature matrix
    let mut result = FeatureMatrix::new(input_dim);
    for (i, symbol) in symbols.iter().enumerate() {
        result.set(symbol, current[i].clone());
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_data() -> (MarketGraph, FeatureMatrix) {
        let mut graph = MarketGraph::new();
        graph.add_edge("BTC", "ETH", 0.8);
        graph.add_edge("ETH", "SOL", 0.7);
        graph.add_edge("BTC", "SOL", 0.6);

        let mut features = FeatureMatrix::new(3);
        features.set("BTC", vec![0.1, 0.2, 0.3]);
        features.set("ETH", vec![0.2, 0.3, 0.4]);
        features.set("SOL", vec![0.3, 0.4, 0.5]);

        (graph, features)
    }

    #[test]
    fn test_gcn_layer() {
        let layer = GraphConvLayer::new(3, 2, Activation::ReLU);

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
        assert_eq!(output[0].len(), 2);
    }

    #[test]
    fn test_gcn() {
        let (graph, features) = create_test_data();

        let gcn = GraphConvolution::new(3, &[4], 1);
        let output = gcn.forward(&graph, &features);

        assert!(output.predictions.contains_key("BTC"));
        assert!(output.predictions.contains_key("ETH"));
        assert!(output.predictions.contains_key("SOL"));
    }

    #[test]
    fn test_gcn_features() {
        let (graph, features) = create_test_data();

        let smoothed = gcn_features(&graph, &features, 2);

        // Features should be aggregated from neighbors
        assert!(smoothed.get("BTC").is_some());
    }
}
