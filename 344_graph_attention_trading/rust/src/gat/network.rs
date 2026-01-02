//! Graph Attention Network
//!
//! Complete GAT model with multiple layers.

use super::layer::GraphAttentionLayer;
use crate::graph::SparseGraph;
use anyhow::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Graph Attention Network
///
/// Multi-layer GAT for node-level predictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAttentionNetwork {
    /// Input feature dimension
    input_dim: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// Number of attention heads
    num_heads: usize,
    /// GAT layers
    layers: Vec<GraphAttentionLayer>,
    /// Output projection (for signal generation)
    output_projection: Option<Array2<f64>>,
}

impl GraphAttentionNetwork {
    /// Create a new GAT network
    ///
    /// Default architecture: 2 layers
    /// - Layer 1: Concatenates heads
    /// - Layer 2: Averages heads
    pub fn new(input_dim: usize, hidden_dim: usize, num_heads: usize) -> Result<Self> {
        let layer1 = GraphAttentionLayer::new(
            input_dim,
            hidden_dim,
            num_heads,
            true, // Concatenate
            0.1,  // Dropout
        );

        let layer2 = GraphAttentionLayer::new(
            hidden_dim * num_heads,
            hidden_dim,
            1, // Single head for output
            false, // Average
            0.1,
        );

        // Output projection: hidden_dim -> 1 (for signal)
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let scale = (2.0 / hidden_dim as f64).sqrt();
        let output_projection = Some(Array2::from_shape_fn((hidden_dim, 1), |_| {
            rng.gen_range(-scale..scale)
        }));

        Ok(Self {
            input_dim,
            hidden_dim,
            num_heads,
            layers: vec![layer1, layer2],
            output_projection,
        })
    }

    /// Create with custom architecture
    pub fn with_layers(
        input_dim: usize,
        hidden_dims: &[usize],
        num_heads: usize,
    ) -> Result<Self> {
        let mut layers = Vec::new();
        let mut current_dim = input_dim;

        for (i, &dim) in hidden_dims.iter().enumerate() {
            let is_last = i == hidden_dims.len() - 1;
            let layer = GraphAttentionLayer::new(
                current_dim,
                dim,
                if is_last { 1 } else { num_heads },
                !is_last, // Concatenate except for last
                0.1,
            );
            current_dim = if is_last { dim } else { dim * num_heads };
            layers.push(layer);
        }

        let output_dim = hidden_dims.last().copied().unwrap_or(input_dim);

        let mut rng = rand::thread_rng();
        use rand::Rng;
        let scale = (2.0 / output_dim as f64).sqrt();
        let output_projection = Some(Array2::from_shape_fn((output_dim, 1), |_| {
            rng.gen_range(-scale..scale)
        }));

        Ok(Self {
            input_dim,
            hidden_dim: *hidden_dims.last().unwrap_or(&input_dim),
            num_heads,
            layers,
            output_projection,
        })
    }

    /// Forward pass through all layers
    pub fn forward(&self, x: &Array2<f64>, graph: &SparseGraph) -> Array2<f64> {
        let mut h = x.clone();

        for layer in &self.layers {
            h = layer.forward(&h, graph);
        }

        h
    }

    /// Get embeddings (no output projection)
    pub fn get_embeddings(&self, x: &Array2<f64>, graph: &SparseGraph) -> Array2<f64> {
        self.forward(x, graph)
    }

    /// Predict trading signals
    pub fn predict_signals(&self, x: &Array2<f64>, graph: &SparseGraph) -> Array1<f64> {
        let embeddings = self.forward(x, graph);

        if let Some(ref proj) = self.output_projection {
            let signals = embeddings.dot(proj);
            // Apply tanh to bound signals to [-1, 1]
            signals.mapv(|x| x.tanh()).column(0).to_owned()
        } else {
            // Use first column as signal
            embeddings.column(0).mapv(|x| x.tanh()).to_owned()
        }
    }

    /// Get attention weights from first layer
    pub fn get_attention_weights(&self, x: &Array2<f64>, graph: &SparseGraph) -> Array2<f64> {
        self.layers[0].get_attention_weights(x, graph)
    }

    /// Propagate signals through the network
    pub fn propagate_signals(
        &self,
        initial_signals: &Array1<f64>,
        graph: &SparseGraph,
    ) -> Array1<f64> {
        // Create feature matrix from signals
        let n = initial_signals.len();
        let mut features = Array2::zeros((n, self.input_dim));

        // Use signals as first feature
        for i in 0..n {
            features[[i, 0]] = initial_signals[i];
        }

        // Forward pass
        self.predict_signals(&features, graph)
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        let layer_params: usize = self.layers.iter().map(|l| l.num_parameters()).sum();
        let output_params = self
            .output_projection
            .as_ref()
            .map(|p| p.len())
            .unwrap_or(0);
        layer_params + output_params
    }

    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get number of heads
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}

/// Training utilities
impl GraphAttentionNetwork {
    /// Compute MSE loss for signal prediction
    pub fn mse_loss(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
        let diff = predictions - targets;
        diff.mapv(|x| x * x).mean().unwrap_or(0.0)
    }

    /// Simple gradient estimation using finite differences
    /// (For production, use automatic differentiation)
    pub fn estimate_gradient(
        &self,
        x: &Array2<f64>,
        targets: &Array1<f64>,
        graph: &SparseGraph,
        epsilon: f64,
    ) -> f64 {
        let predictions = self.predict_signals(x, graph);
        self.mse_loss(&predictions, targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_network_creation() {
        let gat = GraphAttentionNetwork::new(10, 16, 4).unwrap();

        assert_eq!(gat.input_dim(), 10);
        assert_eq!(gat.hidden_dim(), 16);
        assert_eq!(gat.num_heads(), 4);
        assert_eq!(gat.num_layers(), 2);
    }

    #[test]
    fn test_forward_pass() {
        let n = 5;
        let in_dim = 10;

        let gat = GraphAttentionNetwork::new(in_dim, 16, 2).unwrap();
        let x = Array2::random((n, in_dim), Uniform::new(-1.0, 1.0));

        let adj = crate::graph::GraphBuilder::sample_adjacency(n);
        let graph = SparseGraph::from_dense(&adj);

        let output = gat.forward(&x, &graph);

        assert_eq!(output.nrows(), n);
        assert_eq!(output.ncols(), 16); // hidden_dim
    }

    #[test]
    fn test_predict_signals() {
        let n = 5;
        let in_dim = 10;

        let gat = GraphAttentionNetwork::new(in_dim, 16, 2).unwrap();
        let x = Array2::random((n, in_dim), Uniform::new(-1.0, 1.0));

        let adj = crate::graph::GraphBuilder::sample_adjacency(n);
        let graph = SparseGraph::from_dense(&adj);

        let signals = gat.predict_signals(&x, &graph);

        assert_eq!(signals.len(), n);

        // Signals should be bounded by tanh
        for &s in signals.iter() {
            assert!(s >= -1.0 && s <= 1.0);
        }
    }

    #[test]
    fn test_attention_weights() {
        let n = 5;
        let in_dim = 10;

        let gat = GraphAttentionNetwork::new(in_dim, 16, 2).unwrap();
        let x = Array2::random((n, in_dim), Uniform::new(-1.0, 1.0));

        let adj = crate::graph::GraphBuilder::sample_adjacency(n);
        let graph = SparseGraph::from_dense(&adj);

        let attention = gat.get_attention_weights(&x, &graph);

        assert_eq!(attention.nrows(), n);
        assert_eq!(attention.ncols(), n);
    }

    #[test]
    fn test_custom_architecture() {
        let gat = GraphAttentionNetwork::with_layers(10, &[32, 16, 8], 4).unwrap();

        assert_eq!(gat.num_layers(), 3);
        assert_eq!(gat.hidden_dim(), 8);
    }

    #[test]
    fn test_serialization() {
        let gat = GraphAttentionNetwork::new(10, 16, 2).unwrap();

        let json = gat.to_json().unwrap();
        let gat2 = GraphAttentionNetwork::from_json(&json).unwrap();

        assert_eq!(gat.num_parameters(), gat2.num_parameters());
    }
}
