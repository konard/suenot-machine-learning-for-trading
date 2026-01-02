//! Message Passing Neural Network implementation.
//!
//! This module provides MPNN architectures for learning on market graphs:
//! - Graph Convolutional Networks (GCN)
//! - Graph Attention Networks (GAT)
//! - GraphSAGE
//! - Edge-conditioned convolutions

mod message;
mod aggregate;
mod update;

pub use message::*;
pub use aggregate::*;
pub use update::*;

use crate::graph::MarketGraph;
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during MPNN operations.
#[derive(Error, Debug)]
pub enum MPNNError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Empty graph")]
    EmptyGraph,

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Numerical error: {0}")]
    NumericalError(String),
}

/// Aggregation types for message passing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationType {
    /// Sum all incoming messages
    Sum,
    /// Average all incoming messages
    Mean,
    /// Take the maximum along each dimension
    Max,
    /// Attention-weighted aggregation
    Attention,
}

/// Configuration for MPNN model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MPNNConfig {
    /// Input feature dimension
    pub input_dim: usize,
    /// Hidden layer dimension
    pub hidden_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Number of message passing layers
    pub num_layers: usize,
    /// Aggregation type
    pub aggregation: AggregationType,
    /// Dropout rate (0.0 = no dropout)
    pub dropout: f64,
    /// Whether to use edge features
    pub use_edge_features: bool,
    /// Number of attention heads (for GAT)
    pub num_heads: usize,
}

impl Default for MPNNConfig {
    fn default() -> Self {
        Self {
            input_dim: 20,
            hidden_dim: 64,
            output_dim: 32,
            num_layers: 3,
            aggregation: AggregationType::Mean,
            dropout: 0.1,
            use_edge_features: false,
            num_heads: 4,
        }
    }
}

/// A Message Passing Neural Network layer.
#[derive(Debug, Clone)]
pub struct MPNNLayer {
    /// Weight matrix for node transformation
    pub weights: Array2<f64>,
    /// Bias vector
    pub bias: Array1<f64>,
    /// Aggregation type
    pub aggregation: AggregationType,
    /// Attention weights (for GAT)
    pub attention_weights: Option<Array1<f64>>,
}

impl MPNNLayer {
    /// Create a new MPNN layer with random initialization.
    pub fn new(input_dim: usize, output_dim: usize, aggregation: AggregationType) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let normal = Normal::new(0.0, scale).unwrap();

        let weights = Array2::from_shape_fn((input_dim, output_dim), |_| normal.sample(&mut rng));
        let bias = Array1::zeros(output_dim);

        let attention_weights = if aggregation == AggregationType::Attention {
            Some(Array1::from_shape_fn(2 * output_dim, |_| normal.sample(&mut rng)))
        } else {
            None
        };

        Self {
            weights,
            bias,
            aggregation,
            attention_weights,
        }
    }

    /// Forward pass through the layer.
    pub fn forward(&self, features: &Array2<f64>, adjacency: &Array2<f64>) -> Array2<f64> {
        let n = features.nrows();
        let out_dim = self.weights.ncols();

        // Transform features: H' = H * W
        let transformed = features.dot(&self.weights);

        // Aggregate neighbor features based on adjacency
        let aggregated = match self.aggregation {
            AggregationType::Sum => {
                adjacency.dot(&transformed)
            }
            AggregationType::Mean => {
                let degrees = adjacency.sum_axis(Axis(1));
                let mut result = adjacency.dot(&transformed);
                for i in 0..n {
                    let d = degrees[i].max(1.0);
                    for j in 0..out_dim {
                        result[[i, j]] /= d;
                    }
                }
                result
            }
            AggregationType::Max => {
                let mut result = Array2::zeros((n, out_dim));
                for i in 0..n {
                    for j in 0..out_dim {
                        let mut max_val = f64::NEG_INFINITY;
                        for k in 0..n {
                            if adjacency[[i, k]] > 0.0 {
                                max_val = max_val.max(transformed[[k, j]]);
                            }
                        }
                        result[[i, j]] = if max_val.is_finite() { max_val } else { 0.0 };
                    }
                }
                result
            }
            AggregationType::Attention => {
                self.attention_aggregate(&transformed, adjacency)
            }
        };

        // Add bias and apply activation
        let mut output = aggregated;
        for i in 0..n {
            for j in 0..out_dim {
                output[[i, j]] += self.bias[j];
                // LeakyReLU activation
                if output[[i, j]] < 0.0 {
                    output[[i, j]] *= 0.01;
                }
            }
        }

        output
    }

    /// Attention-based aggregation (GAT-style).
    fn attention_aggregate(&self, features: &Array2<f64>, adjacency: &Array2<f64>) -> Array2<f64> {
        let n = features.nrows();
        let d = features.ncols();

        let att = self.attention_weights.as_ref().unwrap();

        // Compute attention scores
        let mut attention_scores = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if adjacency[[i, j]] > 0.0 {
                    // Concatenate features and compute attention
                    let mut concat = Vec::with_capacity(2 * d);
                    concat.extend(features.row(i).iter());
                    concat.extend(features.row(j).iter());

                    let score: f64 = concat.iter()
                        .zip(att.iter())
                        .map(|(a, b)| a * b)
                        .sum();

                    // LeakyReLU
                    attention_scores[[i, j]] = if score < 0.0 { 0.01 * score } else { score };
                }
            }
        }

        // Softmax normalization per row
        for i in 0..n {
            let max_score = attention_scores.row(i)
                .iter()
                .filter(|&&s| adjacency.row(i).iter().any(|&a| a > 0.0))
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);

            let mut exp_sum = 0.0;
            for j in 0..n {
                if adjacency[[i, j]] > 0.0 {
                    let exp = (attention_scores[[i, j]] - max_score).exp();
                    attention_scores[[i, j]] = exp;
                    exp_sum += exp;
                }
            }

            if exp_sum > 0.0 {
                for j in 0..n {
                    attention_scores[[i, j]] /= exp_sum;
                }
            }
        }

        // Aggregate with attention weights
        attention_scores.dot(features)
    }
}

/// Complete MPNN model.
#[derive(Debug, Clone)]
pub struct MPNN {
    /// Configuration
    pub config: MPNNConfig,
    /// Layers
    pub layers: Vec<MPNNLayer>,
    /// Output projection
    pub output_layer: Array2<f64>,
}

impl MPNN {
    /// Create a new MPNN with the given dimensions.
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, num_layers: usize) -> Self {
        let config = MPNNConfig {
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            ..Default::default()
        };
        Self::from_config(config)
    }

    /// Create an MPNN from configuration.
    pub fn from_config(config: MPNNConfig) -> Self {
        let mut layers = Vec::with_capacity(config.num_layers);

        // First layer: input_dim -> hidden_dim
        layers.push(MPNNLayer::new(
            config.input_dim,
            config.hidden_dim,
            config.aggregation,
        ));

        // Hidden layers: hidden_dim -> hidden_dim
        for _ in 1..config.num_layers {
            layers.push(MPNNLayer::new(
                config.hidden_dim,
                config.hidden_dim,
                config.aggregation,
            ));
        }

        // Output projection
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (config.hidden_dim + config.output_dim) as f64).sqrt();
        let normal = Normal::new(0.0, scale).unwrap();
        let output_layer = Array2::from_shape_fn(
            (config.hidden_dim, config.output_dim),
            |_| normal.sample(&mut rng),
        );

        Self {
            config,
            layers,
            output_layer,
        }
    }

    /// Forward pass through the MPNN.
    pub fn forward(&self, graph: &mut MarketGraph) -> Result<Array2<f64>, MPNNError> {
        if graph.node_count() == 0 {
            return Err(MPNNError::EmptyGraph);
        }

        let adjacency = graph.normalized_adjacency();
        let mut features = graph.feature_matrix();

        // Check dimensions
        if features.ncols() != self.config.input_dim {
            return Err(MPNNError::DimensionMismatch {
                expected: self.config.input_dim,
                actual: features.ncols(),
            });
        }

        // Apply MPNN layers
        for layer in &self.layers {
            features = layer.forward(&features, &adjacency);
        }

        // Output projection
        let output = features.dot(&self.output_layer);

        Ok(output)
    }

    /// Get node embeddings (before output projection).
    pub fn get_embeddings(&self, graph: &mut MarketGraph) -> Result<Array2<f64>, MPNNError> {
        if graph.node_count() == 0 {
            return Err(MPNNError::EmptyGraph);
        }

        let adjacency = graph.normalized_adjacency();
        let mut features = graph.feature_matrix();

        for layer in &self.layers {
            features = layer.forward(&features, &adjacency);
        }

        Ok(features)
    }

    /// Compute a graph-level representation by pooling node embeddings.
    pub fn graph_embedding(&self, graph: &mut MarketGraph) -> Result<Array1<f64>, MPNNError> {
        let embeddings = self.get_embeddings(graph)?;

        // Mean pooling
        let n = embeddings.nrows() as f64;
        let pooled = embeddings.sum_axis(Axis(0)) / n;

        Ok(pooled)
    }

    /// Get attention scores for interpretability (if using attention).
    pub fn get_attention_scores(
        &self,
        graph: &mut MarketGraph,
    ) -> Result<Option<Array2<f64>>, MPNNError> {
        if self.config.aggregation != AggregationType::Attention {
            return Ok(None);
        }

        let adjacency = graph.normalized_adjacency();
        let n = graph.node_count();

        // Get final layer attention
        let layer = self.layers.last().unwrap();

        if layer.attention_weights.is_none() {
            return Ok(None);
        }

        let features = graph.feature_matrix();
        let att = layer.attention_weights.as_ref().unwrap();
        let d = layer.weights.ncols();

        let transformed = features.dot(&layer.weights);
        let mut attention_scores = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if adjacency[[i, j]] > 0.0 {
                    let mut concat = Vec::with_capacity(2 * d);
                    concat.extend(transformed.row(i).iter());
                    concat.extend(transformed.row(j).iter());

                    let score: f64 = concat.iter()
                        .zip(att.iter())
                        .map(|(a, b)| a * b)
                        .sum();

                    attention_scores[[i, j]] = score;
                }
            }
        }

        Ok(Some(attention_scores))
    }
}

/// GraphSAGE-style MPNN with sampling.
#[derive(Debug, Clone)]
pub struct GraphSAGE {
    /// Base MPNN
    pub mpnn: MPNN,
    /// Sample size for each layer
    pub sample_sizes: Vec<usize>,
}

impl GraphSAGE {
    /// Create a new GraphSAGE model.
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        num_layers: usize,
        sample_size: usize,
    ) -> Self {
        let mpnn = MPNN::new(input_dim, hidden_dim, output_dim, num_layers);
        let sample_sizes = vec![sample_size; num_layers];

        Self { mpnn, sample_sizes }
    }

    /// Forward pass with neighborhood sampling.
    pub fn forward(&self, graph: &mut MarketGraph) -> Result<Array2<f64>, MPNNError> {
        // For simplicity, we use the same forward as MPNN
        // In production, you'd implement actual sampling here
        self.mpnn.forward(graph)
    }

    /// Sample neighbors for a node.
    pub fn sample_neighbors(&self, graph: &MarketGraph, node_id: usize, k: usize) -> Vec<usize> {
        let neighbors = graph.neighbors(node_id);
        let mut rng = rand::thread_rng();

        if neighbors.len() <= k {
            neighbors.into_iter().map(|(id, _)| id).collect()
        } else {
            let mut sampled = Vec::with_capacity(k);
            let mut indices: Vec<usize> = (0..neighbors.len()).collect();

            for i in 0..k {
                let j = rng.gen_range(i..indices.len());
                indices.swap(i, j);
                sampled.push(neighbors[indices[i]].0);
            }

            sampled
        }
    }
}

/// Edge-conditioned convolution layer.
#[derive(Debug, Clone)]
pub struct EdgeConditionedConv {
    /// Weight network for edge features
    pub edge_network: Vec<Array2<f64>>,
    /// Output dimension
    pub output_dim: usize,
}

impl EdgeConditionedConv {
    /// Create a new edge-conditioned convolution.
    pub fn new(edge_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        let edge_network = vec![
            Array2::from_shape_fn((edge_dim, hidden_dim), |_| normal.sample(&mut rng)),
            Array2::from_shape_fn((hidden_dim, output_dim * output_dim), |_| normal.sample(&mut rng)),
        ];

        Self {
            edge_network,
            output_dim,
        }
    }

    /// Forward pass with edge features.
    pub fn forward(
        &self,
        node_features: &Array2<f64>,
        edge_features: &Array2<f64>,
        edge_index: &[(usize, usize)],
    ) -> Array2<f64> {
        let n = node_features.nrows();
        let mut output = Array2::zeros((n, self.output_dim));

        for (idx, &(src, tgt)) in edge_index.iter().enumerate() {
            if idx >= edge_features.nrows() {
                continue;
            }

            // Get edge feature
            let edge_feat = edge_features.row(idx);

            // Compute edge-specific weights
            let hidden = edge_feat.to_owned().dot(&self.edge_network[0]);
            let hidden_relu: Array1<f64> = hidden.mapv(|x| x.max(0.0));
            let flat_weights = hidden_relu.dot(&self.edge_network[1]);

            // Reshape to weight matrix and apply
            if let Ok(weights) = flat_weights.into_shape((self.output_dim, self.output_dim)) {
                let src_feat = node_features.row(src).to_owned();
                let transformed = src_feat.dot(&weights);

                for j in 0..self.output_dim {
                    output[[tgt, j]] += transformed[j];
                }
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn create_test_graph() -> MarketGraph {
        let mut graph = MarketGraph::new();

        // Add 4 nodes with 8-dimensional features
        for i in 0..4 {
            let features = Array1::from_vec(vec![i as f64 * 0.1; 8]);
            graph.add_node(format!("NODE{}", i), features);
        }

        // Add edges
        graph.add_edge(0, 1, 0.8);
        graph.add_edge(0, 2, 0.6);
        graph.add_edge(1, 2, 0.7);
        graph.add_edge(2, 3, 0.5);

        graph
    }

    #[test]
    fn test_mpnn_forward() {
        let mut graph = create_test_graph();
        let mpnn = MPNN::new(8, 16, 4, 2);

        let output = mpnn.forward(&mut graph).unwrap();

        assert_eq!(output.nrows(), 4);
        assert_eq!(output.ncols(), 4);
    }

    #[test]
    fn test_mpnn_embeddings() {
        let mut graph = create_test_graph();
        let mpnn = MPNN::new(8, 16, 4, 2);

        let embeddings = mpnn.get_embeddings(&mut graph).unwrap();

        assert_eq!(embeddings.nrows(), 4);
        assert_eq!(embeddings.ncols(), 16);
    }

    #[test]
    fn test_graph_embedding() {
        let mut graph = create_test_graph();
        let mpnn = MPNN::new(8, 16, 4, 2);

        let embedding = mpnn.graph_embedding(&mut graph).unwrap();

        assert_eq!(embedding.len(), 16);
    }

    #[test]
    fn test_attention_aggregation() {
        let mut graph = create_test_graph();

        let config = MPNNConfig {
            input_dim: 8,
            hidden_dim: 16,
            output_dim: 4,
            num_layers: 2,
            aggregation: AggregationType::Attention,
            ..Default::default()
        };

        let mpnn = MPNN::from_config(config);
        let output = mpnn.forward(&mut graph).unwrap();

        assert_eq!(output.nrows(), 4);
    }
}
