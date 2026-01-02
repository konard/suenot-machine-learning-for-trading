//! GNN module for graph neural network layers and models
//!
//! This module provides the core GNN components for learning
//! on dynamic graphs.

mod layers;
mod attention;
mod temporal;

pub use layers::{GNNLayer, GraphConvLayer, MessagePassingLayer};
pub use attention::{GraphAttention, MultiHeadAttention};
pub use temporal::{TemporalMemory, TimeEncoder};

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// GNN model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GNNConfig {
    /// Input feature dimension
    pub input_dim: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Output dimension
    pub output_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Whether to use temporal encoding
    pub use_temporal: bool,
    /// Temporal memory size
    pub memory_size: usize,
    /// Learning rate
    pub learning_rate: f64,
}

impl Default for GNNConfig {
    fn default() -> Self {
        Self {
            input_dim: 10,
            hidden_dims: vec![64, 32],
            output_dim: 16,
            num_heads: 4,
            dropout: 0.1,
            use_temporal: true,
            memory_size: 100,
            learning_rate: 0.001,
        }
    }
}

/// Dynamic GNN model for trading
#[derive(Debug)]
pub struct DynamicGNN {
    /// Model configuration
    pub config: GNNConfig,
    /// GNN layers
    layers: Vec<GNNLayer>,
    /// Attention mechanism
    attention: GraphAttention,
    /// Temporal memory (if enabled)
    temporal_memory: Option<TemporalMemory>,
    /// Time encoder
    time_encoder: TimeEncoder,
    /// Output projection weights
    output_weights: Array2<f64>,
    /// Output bias
    output_bias: Array1<f64>,
}

impl DynamicGNN {
    /// Create a new Dynamic GNN model
    pub fn new(config: GNNConfig) -> Self {
        let mut layers = Vec::new();
        let mut prev_dim = config.input_dim;

        // Build hidden layers
        for &hidden_dim in &config.hidden_dims {
            layers.push(GNNLayer::new(prev_dim, hidden_dim));
            prev_dim = hidden_dim;
        }

        // Attention mechanism
        let attention = GraphAttention::new(prev_dim, config.num_heads);

        // Temporal memory
        let temporal_memory = if config.use_temporal {
            Some(TemporalMemory::new(prev_dim, config.memory_size))
        } else {
            None
        };

        // Time encoder
        let time_encoder = TimeEncoder::new(prev_dim);

        // Output projection
        let output_weights = Array2::from_shape_fn((prev_dim, config.output_dim), |_| {
            rand::random::<f64>() * 0.1 - 0.05
        });
        let output_bias = Array1::zeros(config.output_dim);

        Self {
            config,
            layers,
            attention,
            temporal_memory,
            time_encoder,
            output_weights,
            output_bias,
        }
    }

    /// Forward pass through the GNN
    pub fn forward(
        &mut self,
        features: &Array2<f64>,
        adjacency: &Array2<f64>,
        timestamps: Option<&Array1<f64>>,
    ) -> Array2<f64> {
        let mut h = features.clone();

        // Apply GNN layers
        for layer in &self.layers {
            h = layer.forward(&h, adjacency);
        }

        // Apply attention
        h = self.attention.forward(&h, adjacency);

        // Apply temporal encoding if timestamps provided
        if let Some(ts) = timestamps {
            let time_encoding = self.time_encoder.encode(ts);
            h = &h + &time_encoding;
        }

        // Update temporal memory
        if let Some(ref mut memory) = self.temporal_memory {
            h = memory.update(&h);
        }

        // Output projection
        let output = h.dot(&self.output_weights) + &self.output_bias;

        output
    }

    /// Get node embeddings
    pub fn get_embeddings(
        &mut self,
        features: &Array2<f64>,
        adjacency: &Array2<f64>,
    ) -> Array2<f64> {
        let mut h = features.clone();

        for layer in &self.layers {
            h = layer.forward(&h, adjacency);
        }

        h = self.attention.forward(&h, adjacency);

        h
    }

    /// Predict edge existence probability
    pub fn predict_edge(&self, embedding_i: &Array1<f64>, embedding_j: &Array1<f64>) -> f64 {
        // Dot product similarity
        let dot: f64 = embedding_i.iter().zip(embedding_j.iter()).map(|(a, b)| a * b).sum();
        sigmoid(dot)
    }

    /// Predict price direction
    pub fn predict_direction(&self, embedding: &Array1<f64>) -> (f64, f64, f64) {
        // Simple linear projection to 3 classes (down, neutral, up)
        let sum: f64 = embedding.iter().sum();
        let score = sigmoid(sum / embedding.len() as f64);

        // Convert to probabilities
        let up = score;
        let down = 1.0 - score;
        let neutral = 1.0 - (up - 0.5).abs() * 2.0;

        // Normalize
        let total = up + down + neutral;
        (down / total, neutral / total, up / total)
    }

    /// Get model parameters count
    pub fn param_count(&self) -> usize {
        let mut count = 0;
        for layer in &self.layers {
            count += layer.param_count();
        }
        count += self.attention.param_count();
        count += self.output_weights.len() + self.output_bias.len();
        count
    }
}

/// Sigmoid activation function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// ReLU activation function
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Leaky ReLU activation function
pub fn leaky_relu(x: f64, alpha: f64) -> f64 {
    if x > 0.0 { x } else { alpha * x }
}

/// Softmax for a vector
pub fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Array1<f64> = x.mapv(|v| (v - max).exp());
    let sum: f64 = exp.sum();
    exp / sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gnn_creation() {
        let config = GNNConfig::default();
        let gnn = DynamicGNN::new(config);
        assert!(gnn.param_count() > 0);
    }

    #[test]
    fn test_forward_pass() {
        let config = GNNConfig {
            input_dim: 4,
            hidden_dims: vec![8],
            output_dim: 2,
            ..Default::default()
        };
        let mut gnn = DynamicGNN::new(config);

        let features = Array2::from_shape_fn((3, 4), |_| rand::random::<f64>());
        let adjacency = Array2::from_shape_fn((3, 3), |(i, j)| {
            if i == j { 0.0 } else { 0.5 }
        });

        let output = gnn.forward(&features, &adjacency, None);
        assert_eq!(output.shape(), &[3, 2]);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 0.001);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_softmax() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let s = softmax(&x);
        assert!((s.sum() - 1.0).abs() < 0.001);
    }
}
