//! Equivariant GNN Network
//!
//! Complete E-GNN trading model with multiple layers and prediction heads.

use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::Normal;

use super::config::EGNNConfig;
use super::graph::Graph;
use super::layer::EGNNLayer;

/// Equivariant GNN Trading Model
#[derive(Debug)]
pub struct EquivariantGNN {
    /// Configuration
    config: EGNNConfig,

    /// Input embedding layer
    input_embed: Array2<f64>,
    input_bias: Array1<f64>,

    /// E-GNN layers
    layers: Vec<EGNNLayer>,

    /// Direction prediction head
    direction_w1: Array2<f64>,
    direction_b1: Array1<f64>,
    direction_w2: Array2<f64>,
    direction_b2: Array1<f64>,

    /// Position sizing head
    sizing_w: Array2<f64>,
    sizing_b: Array1<f64>,

    /// Risk prediction head
    risk_w: Array2<f64>,
    risk_b: Array1<f64>,
}

/// Model output
#[derive(Debug, Clone)]
pub struct ModelOutput {
    /// Direction probabilities [num_nodes, 3] (Short, Hold, Long)
    pub direction_probs: Array2<f64>,

    /// Position sizes [num_nodes]
    pub position_sizes: Array1<f64>,

    /// Volatility predictions [num_nodes]
    pub volatility: Array1<f64>,

    /// VaR predictions [num_nodes]
    pub var: Array1<f64>,

    /// Updated coordinates
    pub coordinates: Array2<f64>,
}

impl EquivariantGNN {
    /// Create a new E-GNN model
    pub fn new(input_dim: usize, hidden_dim: usize, coord_dim: usize, num_layers: usize) -> Self {
        let config = EGNNConfig {
            input_dim,
            hidden_dim,
            coord_dim,
            num_layers,
            ..Default::default()
        };

        Self::from_config(config)
    }

    /// Create from configuration
    pub fn from_config(config: EGNNConfig) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        // Input embedding
        let input_embed = Array2::from_shape_fn(
            (config.input_dim, config.hidden_dim),
            |_| rng.sample(normal),
        );
        let input_bias = Array1::zeros(config.hidden_dim);

        // E-GNN layers
        let layers: Vec<EGNNLayer> = (0..config.num_layers)
            .map(|i| {
                EGNNLayer::new(
                    config.hidden_dim,
                    config.edge_dim,
                    config.update_coords && i < config.num_layers - 1,
                )
            })
            .collect();

        // Direction head
        let direction_w1 = Array2::from_shape_fn(
            (config.hidden_dim, config.hidden_dim / 2),
            |_| rng.sample(normal),
        );
        let direction_b1 = Array1::zeros(config.hidden_dim / 2);
        let direction_w2 = Array2::from_shape_fn(
            (config.hidden_dim / 2, config.output_classes),
            |_| rng.sample(normal),
        );
        let direction_b2 = Array1::zeros(config.output_classes);

        // Sizing head
        let sizing_w = Array2::from_shape_fn((config.hidden_dim, 1), |_| rng.sample(normal));
        let sizing_b = Array1::zeros(1);

        // Risk head
        let risk_w = Array2::from_shape_fn((config.hidden_dim, 2), |_| rng.sample(normal));
        let risk_b = Array1::zeros(2);

        Self {
            config,
            input_embed,
            input_bias,
            layers,
            direction_w1,
            direction_b1,
            direction_w2,
            direction_b2,
            sizing_w,
            sizing_b,
            risk_w,
            risk_b,
        }
    }

    /// Forward pass through the model
    pub fn forward(&self, graph: &Graph) -> ModelOutput {
        let num_nodes = graph.num_nodes();

        // Embed input features
        let mut h = self.embed_input(&graph.node_features);
        let mut x = graph.coordinates.clone();

        // Pass through E-GNN layers
        for layer in &self.layers {
            let (h_new, x_new) = layer.forward(
                &h,
                &x,
                &graph.edge_index,
                Some(&graph.edge_features),
            );
            h = h_new;
            x = x_new;
        }

        // Direction predictions
        let direction_probs = self.direction_head(&h);

        // Position sizing
        let position_sizes = self.sizing_head(&h);

        // Risk predictions
        let (volatility, var) = self.risk_head(&h);

        ModelOutput {
            direction_probs,
            position_sizes,
            volatility,
            var,
            coordinates: x,
        }
    }

    /// Predict trading signals
    pub fn predict(&self, graph: &Graph) -> Vec<i32> {
        let output = self.forward(graph);
        self.signals_from_output(&output, 0.4)
    }

    /// Convert output to trading signals
    pub fn signals_from_output(&self, output: &ModelOutput, threshold: f64) -> Vec<i32> {
        let num_nodes = output.direction_probs.nrows();
        let mut signals = Vec::with_capacity(num_nodes);

        for i in 0..num_nodes {
            let short_prob = output.direction_probs[[i, 0]];
            let long_prob = output.direction_probs[[i, 2]];

            if long_prob > threshold {
                signals.push(1); // Long
            } else if short_prob > threshold {
                signals.push(-1); // Short
            } else {
                signals.push(0); // Hold
            }
        }

        signals
    }

    /// Embed input features to hidden dimension
    fn embed_input(&self, features: &Array2<f64>) -> Array2<f64> {
        let num_nodes = features.nrows();
        let mut embedded = Array2::zeros((num_nodes, self.config.hidden_dim));

        for i in 0..num_nodes {
            let input = features.row(i);
            let mut z = Array1::zeros(self.config.hidden_dim);

            // Linear transformation
            for j in 0..self.config.hidden_dim {
                let mut sum = self.input_bias[j];
                for k in 0..self.config.input_dim.min(input.len()) {
                    sum += input[k] * self.input_embed[[k, j]];
                }
                z[j] = relu(sum);
            }

            embedded.row_mut(i).assign(&z);
        }

        embedded
    }

    /// Direction prediction head
    fn direction_head(&self, h: &Array2<f64>) -> Array2<f64> {
        let num_nodes = h.nrows();
        let mut probs = Array2::zeros((num_nodes, self.config.output_classes));

        for i in 0..num_nodes {
            let input = h.row(i).to_owned();

            // First layer
            let z1 = input.dot(&self.direction_w1) + &self.direction_b1;
            let a1 = z1.mapv(relu);

            // Second layer
            let z2 = a1.dot(&self.direction_w2) + &self.direction_b2;

            // Softmax
            let softmax = self.softmax(&z2);
            probs.row_mut(i).assign(&softmax);
        }

        probs
    }

    /// Position sizing head
    fn sizing_head(&self, h: &Array2<f64>) -> Array1<f64> {
        let num_nodes = h.nrows();
        let mut sizes = Array1::zeros(num_nodes);

        for i in 0..num_nodes {
            let input = h.row(i).to_owned();
            let z = input.dot(&self.sizing_w) + &self.sizing_b;
            sizes[i] = sigmoid(z[0]);
        }

        sizes
    }

    /// Risk prediction head
    fn risk_head(&self, h: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
        let num_nodes = h.nrows();
        let mut volatility = Array1::zeros(num_nodes);
        let mut var = Array1::zeros(num_nodes);

        for i in 0..num_nodes {
            let input = h.row(i).to_owned();
            let z = input.dot(&self.risk_w) + &self.risk_b;
            volatility[i] = softplus(z[0]);
            var[i] = z[1];
        }

        (volatility, var)
    }

    /// Softmax activation
    fn softmax(&self, x: &Array1<f64>) -> Array1<f64> {
        let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_x: Array1<f64> = x.mapv(|v| (v - max_val).exp());
        let sum: f64 = exp_x.sum();
        exp_x / sum
    }

    /// Get configuration
    pub fn config(&self) -> &EGNNConfig {
        &self.config
    }
}

/// ReLU activation
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Sigmoid activation
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Softplus activation
fn softplus(x: f64) -> f64 {
    (1.0 + x.exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egnn::graph::{GraphNode, GraphEdge};

    fn create_test_graph() -> Graph {
        let nodes = vec![
            GraphNode::new(0, "BTC".to_string(), vec![0.01; 10], vec![1.0, 0.0, 0.0]),
            GraphNode::new(1, "ETH".to_string(), vec![0.02; 10], vec![0.0, 1.0, 0.0]),
            GraphNode::new(2, "SOL".to_string(), vec![0.015; 10], vec![0.0, 0.0, 1.0]),
        ];

        let edges = vec![
            GraphEdge::new(0, 1, vec![0.8, 0.8, 1.0]),
            GraphEdge::new(1, 0, vec![0.8, 0.8, 1.0]),
            GraphEdge::new(0, 2, vec![0.6, 0.6, 1.0]),
            GraphEdge::new(2, 0, vec![0.6, 0.6, 1.0]),
            GraphEdge::new(1, 2, vec![0.7, 0.7, 1.0]),
            GraphEdge::new(2, 1, vec![0.7, 0.7, 1.0]),
        ];

        Graph::from_nodes_edges(nodes, edges)
    }

    #[test]
    fn test_model_creation() {
        let model = EquivariantGNN::new(10, 64, 3, 4);
        assert_eq!(model.config().hidden_dim, 64);
        assert_eq!(model.config().num_layers, 4);
    }

    #[test]
    fn test_forward_pass() {
        let model = EquivariantGNN::new(10, 32, 3, 2);
        let graph = create_test_graph();

        let output = model.forward(&graph);

        assert_eq!(output.direction_probs.nrows(), 3);
        assert_eq!(output.direction_probs.ncols(), 3);
        assert_eq!(output.position_sizes.len(), 3);

        // Check probabilities sum to 1
        for i in 0..3 {
            let sum: f64 = output.direction_probs.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_predict() {
        let model = EquivariantGNN::new(10, 32, 3, 2);
        let graph = create_test_graph();

        let signals = model.predict(&graph);

        assert_eq!(signals.len(), 3);
        for signal in signals {
            assert!(signal >= -1 && signal <= 1);
        }
    }
}
