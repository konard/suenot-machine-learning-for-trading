//! Graph Transformer model
//!
//! Full Graph Transformer model combining multiple layers with prediction heads.

use super::config::GraphTransformerConfig;
use super::layer::GraphTransformerLayer;
use super::positional::PositionalEncoding;
use crate::graph::market_graph::MarketGraph;
use ndarray::{Array1, Array2};
use rand::Rng;

/// Graph Transformer model for cryptocurrency prediction
#[derive(Debug)]
pub struct GraphTransformer {
    /// Model configuration
    pub config: GraphTransformerConfig,
    /// Input projection
    pub input_proj: Array2<f64>,
    /// Positional encoding
    pub pos_encoding: PositionalEncoding,
    /// Transformer layers
    pub layers: Vec<GraphTransformerLayer>,
    /// Direction prediction head (up/neutral/down)
    pub direction_head: Array2<f64>,
    /// Return prediction head (regression)
    pub return_head: Array2<f64>,
}

impl GraphTransformer {
    /// Create a new Graph Transformer model
    pub fn new(config: GraphTransformerConfig) -> Self {
        config.validate().expect("Invalid configuration");

        let mut rng = rand::thread_rng();

        // Input projection from input_dim + pe_dim to hidden_dim
        let proj_in = config.input_dim + config.num_pe_dims;
        let proj_xavier = (2.0 / (proj_in + config.hidden_dim) as f64).sqrt();
        let input_proj = Self::random_matrix(proj_in, config.hidden_dim, proj_xavier, &mut rng);

        // Positional encoding
        let pos_encoding = PositionalEncoding::new(config.positional_encoding, config.num_pe_dims);

        // Create transformer layers
        let layers: Vec<GraphTransformerLayer> = (0..config.num_layers)
            .map(|_| GraphTransformerLayer::new(&config))
            .collect();

        // Prediction heads
        let head_xavier = (2.0 / (config.hidden_dim + config.output_dim) as f64).sqrt();
        let direction_head = Self::random_matrix(config.hidden_dim, config.output_dim, head_xavier, &mut rng);
        let return_head = Self::random_matrix(config.hidden_dim, 1, head_xavier, &mut rng);

        Self {
            config,
            input_proj,
            pos_encoding,
            layers,
            direction_head,
            return_head,
        }
    }

    fn random_matrix(rows: usize, cols: usize, std: f64, rng: &mut impl Rng) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |_| rng.gen::<f64>() * std * 2.0 - std)
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    /// * `x` - Node features [num_nodes, input_dim]
    /// * `graph` - Market graph structure
    ///
    /// # Returns
    /// Node embeddings [num_nodes, hidden_dim]
    pub fn forward(&self, x: &Array2<f64>, graph: &MarketGraph) -> Array2<f64> {
        // Add positional encoding
        let x_with_pe = self.pos_encoding.add_to_features(x, graph);

        // Project to hidden dimension
        let mut hidden = x_with_pe.dot(&self.input_proj);

        // Get edge index
        let edge_index = graph.edge_index_coo();

        // Compute edge attributes if needed
        let edge_attr: Option<Array2<f64>> = if self.config.use_edge_features {
            if let Some(edge_dim) = self.config.edge_dim {
                let mut attr = Array2::zeros((graph.num_edges(), edge_dim));
                for (i, edge) in graph.edges.iter().enumerate() {
                    let feats = edge.compute_features(edge_dim);
                    attr.row_mut(i).assign(&feats);
                }
                Some(attr)
            } else {
                None
            }
        } else {
            None
        };

        // Pass through transformer layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &edge_index, edge_attr.as_ref());
        }

        hidden
    }

    /// Get direction predictions (up/neutral/down)
    ///
    /// # Returns
    /// Probabilities [num_nodes, 3] after softmax
    pub fn predict_direction(&self, x: &Array2<f64>, graph: &MarketGraph) -> Array2<f64> {
        let embeddings = self.forward(x, graph);
        let logits = embeddings.dot(&self.direction_head);

        // Apply softmax row-wise
        let mut probs = Array2::zeros(logits.raw_dim());
        for i in 0..logits.nrows() {
            let row = logits.row(i);
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = row.iter().map(|&v| (v - max_val).exp()).sum();
            for j in 0..logits.ncols() {
                probs[[i, j]] = (logits[[i, j]] - max_val).exp() / exp_sum;
            }
        }

        probs
    }

    /// Get return predictions (regression)
    ///
    /// # Returns
    /// Predicted returns [num_nodes]
    pub fn predict_return(&self, x: &Array2<f64>, graph: &MarketGraph) -> Array1<f64> {
        let embeddings = self.forward(x, graph);
        let returns = embeddings.dot(&self.return_head);
        returns.column(0).to_owned()
    }

    /// Get both predictions
    pub fn predict(&self, x: &Array2<f64>, graph: &MarketGraph) -> (Array2<f64>, Array1<f64>) {
        let embeddings = self.forward(x, graph);

        // Direction predictions with softmax
        let logits = embeddings.dot(&self.direction_head);
        let mut direction_probs = Array2::zeros(logits.raw_dim());
        for i in 0..logits.nrows() {
            let row = logits.row(i);
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = row.iter().map(|&v| (v - max_val).exp()).sum();
            for j in 0..logits.ncols() {
                direction_probs[[i, j]] = (logits[[i, j]] - max_val).exp() / exp_sum;
            }
        }

        // Return predictions
        let returns = embeddings.dot(&self.return_head).column(0).to_owned();

        (direction_probs, returns)
    }

    /// Get node embeddings for downstream tasks
    pub fn get_embeddings(&self, x: &Array2<f64>, graph: &MarketGraph) -> Array2<f64> {
        self.forward(x, graph)
    }

    /// Get attention weights from all layers
    pub fn get_attention_weights(
        &self,
        x: &Array2<f64>,
        graph: &MarketGraph,
    ) -> Vec<std::collections::HashMap<(usize, usize), f64>> {
        let x_with_pe = self.pos_encoding.add_to_features(x, graph);
        let mut hidden = x_with_pe.dot(&self.input_proj);
        let edge_index = graph.edge_index_coo();

        let mut all_weights = Vec::new();

        for layer in &self.layers {
            let weights = layer.attention.get_attention_weights(&hidden, &edge_index);
            all_weights.push(weights);
            hidden = layer.forward(&hidden, &edge_index, None);
        }

        all_weights
    }
}

/// Prediction result for a single node
#[derive(Debug, Clone)]
pub struct NodePrediction {
    pub symbol: String,
    pub prob_up: f64,
    pub prob_neutral: f64,
    pub prob_down: f64,
    pub predicted_return: f64,
}

impl NodePrediction {
    /// Get the most likely direction
    pub fn predicted_direction(&self) -> &str {
        if self.prob_up > self.prob_down && self.prob_up > self.prob_neutral {
            "UP"
        } else if self.prob_down > self.prob_up && self.prob_down > self.prob_neutral {
            "DOWN"
        } else {
            "NEUTRAL"
        }
    }

    /// Get confidence (max probability)
    pub fn confidence(&self) -> f64 {
        self.prob_up.max(self.prob_neutral).max(self.prob_down)
    }
}

/// Get predictions for all nodes with symbols
pub fn get_predictions(
    model: &GraphTransformer,
    features: &Array2<f64>,
    graph: &MarketGraph,
) -> Vec<NodePrediction> {
    let (direction_probs, returns) = model.predict(features, graph);

    graph
        .nodes
        .iter()
        .enumerate()
        .map(|(i, node)| NodePrediction {
            symbol: node.symbol.clone(),
            prob_up: direction_probs[[i, 0]],
            prob_neutral: direction_probs[[i, 1]],
            prob_down: direction_probs[[i, 2]],
            predicted_return: returns[i],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builder::create_test_graph;

    #[test]
    fn test_model_creation() {
        let config = GraphTransformerConfig::minimal();
        let model = GraphTransformer::new(config);
        assert_eq!(model.layers.len(), 2);
    }

    #[test]
    fn test_forward_pass() {
        let config = GraphTransformerConfig::minimal();
        let model = GraphTransformer::new(config.clone());
        let graph = create_test_graph();

        let x = Array2::from_shape_fn((graph.num_nodes(), config.input_dim), |_| {
            rand::random::<f64>()
        });

        let embeddings = model.forward(&x, &graph);
        assert_eq!(embeddings.nrows(), graph.num_nodes());
        assert_eq!(embeddings.ncols(), config.hidden_dim);
    }

    #[test]
    fn test_predictions() {
        let config = GraphTransformerConfig::minimal();
        let model = GraphTransformer::new(config.clone());
        let graph = create_test_graph();

        let x = Array2::from_shape_fn((graph.num_nodes(), config.input_dim), |_| {
            rand::random::<f64>()
        });

        let (direction, returns) = model.predict(&x, &graph);

        assert_eq!(direction.nrows(), graph.num_nodes());
        assert_eq!(direction.ncols(), 3);
        assert_eq!(returns.len(), graph.num_nodes());

        // Check softmax (rows sum to 1)
        for i in 0..direction.nrows() {
            let sum: f64 = direction.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}
