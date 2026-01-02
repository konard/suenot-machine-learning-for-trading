//! Heterogeneous Graph Neural Network module
//!
//! This module provides GNN layers with type-aware attention for heterogeneous graphs.

mod layers;
mod attention;
mod metapath;
mod aggregation;

pub use layers::{HGNNLayer, LinearLayer, ActivationFn};
pub use attention::{TypeProjection, RelationAttention, MultiHeadAttention};
pub use metapath::MetapathEncoder;
pub use aggregation::SemanticAggregation;

use ndarray::{Array1, Array2};
use std::collections::HashMap;

use crate::graph::{NodeType, EdgeType, HeterogeneousGraph};

/// Configuration for Heterogeneous GNN
#[derive(Debug, Clone)]
pub struct HGNNConfig {
    /// Input dimension per node type
    pub input_dims: HashMap<NodeType, usize>,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Output embedding dimension
    pub output_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Whether to use metapath-based aggregation
    pub use_metapaths: bool,
    /// Number of metapath samples per node
    pub metapath_samples: usize,
    /// Learning rate
    pub learning_rate: f64,
}

impl Default for HGNNConfig {
    fn default() -> Self {
        let mut input_dims = HashMap::new();
        for node_type in NodeType::all() {
            input_dims.insert(node_type, node_type.feature_dim());
        }

        Self {
            input_dims,
            hidden_dims: vec![64, 32],
            output_dim: 16,
            num_heads: 4,
            dropout: 0.1,
            use_metapaths: true,
            metapath_samples: 10,
            learning_rate: 0.001,
        }
    }
}

/// Heterogeneous Graph Neural Network
#[derive(Debug)]
pub struct HeterogeneousGNN {
    /// Configuration
    config: HGNNConfig,
    /// Type-specific projection layers
    type_projections: HashMap<NodeType, TypeProjection>,
    /// HGNN layers
    layers: Vec<HGNNLayer>,
    /// Relation-specific attention
    relation_attention: HashMap<EdgeType, RelationAttention>,
    /// Metapath encoder
    metapath_encoder: Option<MetapathEncoder>,
    /// Semantic aggregation layer
    semantic_aggregation: SemanticAggregation,
    /// Prediction heads
    direction_head: LinearLayer,
    magnitude_head: LinearLayer,
}

impl HeterogeneousGNN {
    /// Create a new Heterogeneous GNN
    pub fn new(config: HGNNConfig) -> Self {
        // Create type-specific projections
        let common_dim = config.hidden_dims.first().copied().unwrap_or(64);
        let mut type_projections = HashMap::new();
        for (node_type, &input_dim) in &config.input_dims {
            type_projections.insert(*node_type, TypeProjection::new(input_dim, common_dim));
        }

        // Create HGNN layers
        let mut layers = Vec::new();
        let mut prev_dim = common_dim;
        for &hidden_dim in &config.hidden_dims {
            layers.push(HGNNLayer::new(prev_dim, hidden_dim, config.num_heads));
            prev_dim = hidden_dim;
        }

        // Create relation-specific attention
        let mut relation_attention = HashMap::new();
        for edge_type in EdgeType::all() {
            relation_attention.insert(
                edge_type,
                RelationAttention::new(prev_dim, config.num_heads),
            );
        }

        // Create metapath encoder if enabled
        let metapath_encoder = if config.use_metapaths {
            Some(MetapathEncoder::new(prev_dim, config.metapath_samples))
        } else {
            None
        };

        // Create semantic aggregation
        let semantic_aggregation = SemanticAggregation::new(prev_dim);

        // Create prediction heads
        let direction_head = LinearLayer::new(config.output_dim, 3);  // up, down, neutral
        let magnitude_head = LinearLayer::new(config.output_dim, 1);  // expected return

        Self {
            config,
            type_projections,
            layers,
            relation_attention,
            metapath_encoder,
            semantic_aggregation,
            direction_head,
            magnitude_head,
        }
    }

    /// Get node embeddings from the graph
    pub fn get_embeddings(
        &self,
        graph: &HeterogeneousGraph,
    ) -> HashMap<String, Array1<f64>> {
        let mut embeddings = HashMap::new();

        // Project all nodes to common space
        for node_type in NodeType::all() {
            let (features, node_ids) = graph.feature_matrix_by_type(node_type);
            if node_ids.is_empty() {
                continue;
            }

            let projection = &self.type_projections[&node_type];
            let projected = projection.forward(&features);

            for (i, node_id) in node_ids.iter().enumerate() {
                embeddings.insert(node_id.clone(), projected.row(i).to_owned());
            }
        }

        // Apply HGNN layers with message passing
        for layer in &self.layers {
            let mut new_embeddings = HashMap::new();

            for (node_id, embedding) in &embeddings {
                let neighbors = graph.get_neighbors(node_id);

                if neighbors.is_empty() {
                    // No neighbors, just apply layer transformation
                    new_embeddings.insert(node_id.clone(), layer.forward_single(embedding));
                } else {
                    // Aggregate neighbor messages by edge type
                    let mut messages_by_type: HashMap<EdgeType, Vec<Array1<f64>>> = HashMap::new();

                    for (neighbor, edge) in neighbors {
                        if let Some(neighbor_emb) = embeddings.get(&neighbor.id) {
                            let attention = &self.relation_attention[&edge.edge_type];
                            let message = attention.compute_message(embedding, neighbor_emb, &edge.features);
                            messages_by_type
                                .entry(edge.edge_type)
                                .or_default()
                                .push(message);
                        }
                    }

                    // Aggregate messages
                    let aggregated = self.aggregate_messages(embedding, &messages_by_type);
                    new_embeddings.insert(node_id.clone(), layer.forward_with_message(embedding, &aggregated));
                }
            }

            embeddings = new_embeddings;
        }

        embeddings
    }

    /// Aggregate messages from different edge types
    fn aggregate_messages(
        &self,
        _node_embedding: &Array1<f64>,
        messages_by_type: &HashMap<EdgeType, Vec<Array1<f64>>>,
    ) -> Array1<f64> {
        if messages_by_type.is_empty() {
            return Array1::zeros(self.config.hidden_dims.last().copied().unwrap_or(32));
        }

        // Average messages within each type
        let type_aggregates: Vec<Array1<f64>> = messages_by_type
            .iter()
            .map(|(_, messages)| {
                let n = messages.len() as f64;
                messages.iter().fold(
                    Array1::zeros(messages[0].len()),
                    |acc, m| acc + m,
                ) / n
            })
            .collect();

        // Semantic aggregation across types
        self.semantic_aggregation.aggregate(&type_aggregates)
    }

    /// Predict price direction
    pub fn predict_direction(&self, embedding: &Array1<f64>) -> (f64, f64, f64) {
        let logits = self.direction_head.forward(embedding);
        let exp_logits: Vec<f64> = logits.iter().map(|x| x.exp()).collect();
        let sum: f64 = exp_logits.iter().sum();

        (
            exp_logits[0] / sum,  // P(down)
            exp_logits[1] / sum,  // P(neutral)
            exp_logits[2] / sum,  // P(up)
        )
    }

    /// Predict expected return magnitude
    pub fn predict_magnitude(&self, embedding: &Array1<f64>) -> f64 {
        let output = self.magnitude_head.forward(embedding);
        output[0].tanh() * 0.1  // Scale to reasonable return range
    }

    /// Predict edge existence probability
    pub fn predict_edge(&self, emb1: &Array1<f64>, emb2: &Array1<f64>) -> f64 {
        // Simple dot product similarity
        let dot: f64 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
        1.0 / (1.0 + (-dot).exp())  // Sigmoid
    }

    /// Get the total number of parameters
    pub fn param_count(&self) -> usize {
        let mut count = 0;

        for projection in self.type_projections.values() {
            count += projection.param_count();
        }

        for layer in &self.layers {
            count += layer.param_count();
        }

        for attention in self.relation_attention.values() {
            count += attention.param_count();
        }

        count += self.direction_head.param_count();
        count += self.magnitude_head.param_count();

        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = HGNNConfig::default();
        assert_eq!(config.num_heads, 4);
        assert!(config.use_metapaths);
    }

    #[test]
    fn test_model_creation() {
        let config = HGNNConfig::default();
        let model = HeterogeneousGNN::new(config);
        assert!(model.param_count() > 0);
    }
}
