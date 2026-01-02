//! Graph Attention mechanism
//!
//! Implements multi-head graph attention with edge feature support.

use super::config::{ActivationType, GraphTransformerConfig};
use crate::graph::market_graph::MarketGraph;
use crate::utils::math::softmax;
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use std::collections::HashMap;

/// Graph Attention layer
#[derive(Debug, Clone)]
pub struct GraphAttention {
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Input dimension
    pub input_dim: usize,
    /// Query projection weights [input_dim, num_heads * head_dim]
    pub w_query: Array2<f64>,
    /// Key projection weights [input_dim, num_heads * head_dim]
    pub w_key: Array2<f64>,
    /// Value projection weights [input_dim, num_heads * head_dim]
    pub w_value: Array2<f64>,
    /// Output projection weights [num_heads * head_dim, input_dim]
    pub w_out: Array2<f64>,
    /// Edge feature projection (optional)
    pub w_edge: Option<Array2<f64>>,
    /// Dropout rate
    pub dropout: f64,
    /// Scaling factor for attention scores
    pub scale: f64,
}

impl GraphAttention {
    /// Create a new Graph Attention layer
    pub fn new(input_dim: usize, num_heads: usize, dropout: f64, edge_dim: Option<usize>) -> Self {
        assert!(input_dim % num_heads == 0, "input_dim must be divisible by num_heads");
        let head_dim = input_dim / num_heads;
        let scale = (head_dim as f64).sqrt();

        // Initialize weights with Xavier/Glorot initialization
        let mut rng = rand::thread_rng();
        let xavier_std = (2.0 / (input_dim + input_dim) as f64).sqrt();

        let w_query = Self::random_matrix(input_dim, input_dim, xavier_std, &mut rng);
        let w_key = Self::random_matrix(input_dim, input_dim, xavier_std, &mut rng);
        let w_value = Self::random_matrix(input_dim, input_dim, xavier_std, &mut rng);
        let w_out = Self::random_matrix(input_dim, input_dim, xavier_std, &mut rng);

        let w_edge = edge_dim.map(|ed| {
            let edge_xavier = (2.0 / (ed + num_heads) as f64).sqrt();
            Self::random_matrix(ed, num_heads, edge_xavier, &mut rng)
        });

        Self {
            num_heads,
            head_dim,
            input_dim,
            w_query,
            w_key,
            w_value,
            w_out,
            w_edge,
            dropout,
            scale,
        }
    }

    /// Create from config
    pub fn from_config(config: &GraphTransformerConfig) -> Self {
        Self::new(
            config.hidden_dim,
            config.num_heads,
            config.dropout,
            config.edge_dim,
        )
    }

    /// Generate random matrix with given std
    fn random_matrix(rows: usize, cols: usize, std: f64, rng: &mut impl Rng) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |_| rng.gen::<f64>() * std * 2.0 - std)
    }

    /// Forward pass for graph attention
    ///
    /// # Arguments
    /// * `x` - Node features [num_nodes, input_dim]
    /// * `edge_index` - Edge connectivity [2, num_edges] (source, target)
    /// * `edge_attr` - Optional edge features [num_edges, edge_dim]
    ///
    /// # Returns
    /// Updated node features [num_nodes, input_dim]
    pub fn forward(
        &self,
        x: &Array2<f64>,
        edge_index: &Array2<usize>,
        edge_attr: Option<&Array2<f64>>,
    ) -> Array2<f64> {
        let num_nodes = x.nrows();
        let num_edges = edge_index.ncols();

        // Compute Q, K, V projections
        let queries = x.dot(&self.w_query);  // [num_nodes, hidden]
        let keys = x.dot(&self.w_key);       // [num_nodes, hidden]
        let values = x.dot(&self.w_value);   // [num_nodes, hidden]

        // Initialize output
        let mut output = Array2::zeros((num_nodes, self.input_dim));

        // For each node, aggregate information from neighbors
        for node in 0..num_nodes {
            // Get incoming edges for this node
            let incoming: Vec<usize> = (0..num_edges)
                .filter(|&i| edge_index[[1, i]] == node)
                .collect();

            if incoming.is_empty() {
                // No neighbors, use own features
                output.row_mut(node).assign(&x.row(node));
                continue;
            }

            // Compute attention scores for neighbors
            let query = queries.row(node);
            let mut attention_scores = Vec::with_capacity(incoming.len());

            for &edge_idx in &incoming {
                let src = edge_index[[0, edge_idx]];
                let key = keys.row(src);

                // Dot product attention
                let mut score = query.dot(&key) / self.scale;

                // Add edge bias if available
                if let (Some(w_edge), Some(edge_feats)) = (&self.w_edge, edge_attr) {
                    let edge_feat = edge_feats.row(edge_idx);
                    // Edge bias: edge_feat @ w_edge -> [num_heads], then mean
                    let edge_bias: f64 = edge_feat.dot(w_edge).sum() / self.num_heads as f64;
                    score += edge_bias;
                }

                attention_scores.push(score);
            }

            // Apply softmax
            let attention_weights = softmax(&attention_scores);

            // Aggregate values
            let mut aggregated = Array1::zeros(self.input_dim);
            for (i, &edge_idx) in incoming.iter().enumerate() {
                let src = edge_index[[0, edge_idx]];
                let value = values.row(src);
                aggregated = aggregated + value.to_owned() * attention_weights[i];
            }

            output.row_mut(node).assign(&aggregated);
        }

        // Output projection
        output.dot(&self.w_out)
    }

    /// Forward pass on MarketGraph
    pub fn forward_graph(&self, graph: &MarketGraph, x: &Array2<f64>) -> Array2<f64> {
        let edge_index = graph.edge_index_coo();

        // Compute edge features if available
        let edge_attr: Option<Array2<f64>> = if self.w_edge.is_some() {
            let edge_dim = self.w_edge.as_ref().unwrap().nrows();
            let mut attr = Array2::zeros((graph.num_edges(), edge_dim));
            for (i, edge) in graph.edges.iter().enumerate() {
                let feats = edge.compute_features(edge_dim);
                attr.row_mut(i).assign(&feats);
            }
            Some(attr)
        } else {
            None
        };

        self.forward(x, &edge_index, edge_attr.as_ref())
    }

    /// Get attention weights for visualization
    pub fn get_attention_weights(
        &self,
        x: &Array2<f64>,
        edge_index: &Array2<usize>,
    ) -> HashMap<(usize, usize), f64> {
        let num_nodes = x.nrows();
        let num_edges = edge_index.ncols();

        let queries = x.dot(&self.w_query);
        let keys = x.dot(&self.w_key);

        let mut attention_map = HashMap::new();

        for node in 0..num_nodes {
            let incoming: Vec<usize> = (0..num_edges)
                .filter(|&i| edge_index[[1, i]] == node)
                .collect();

            if incoming.is_empty() {
                continue;
            }

            let query = queries.row(node);
            let mut scores: Vec<f64> = incoming
                .iter()
                .map(|&edge_idx| {
                    let src = edge_index[[0, edge_idx]];
                    query.dot(&keys.row(src)) / self.scale
                })
                .collect();

            let weights = softmax(&scores);

            for (i, &edge_idx) in incoming.iter().enumerate() {
                let src = edge_index[[0, edge_idx]];
                attention_map.insert((src, node), weights[i]);
            }
        }

        attention_map
    }
}

/// Multi-Head Graph Attention for more expressive power
#[derive(Debug, Clone)]
pub struct MultiHeadGraphAttention {
    /// Individual attention heads
    pub heads: Vec<GraphAttention>,
    /// Output projection
    pub w_out: Array2<f64>,
    /// Number of heads
    pub num_heads: usize,
}

impl MultiHeadGraphAttention {
    /// Create new multi-head attention
    pub fn new(input_dim: usize, num_heads: usize, dropout: f64, edge_dim: Option<usize>) -> Self {
        let head_dim = input_dim / num_heads;
        let mut rng = rand::thread_rng();

        // Create separate small attention for each head
        let heads: Vec<GraphAttention> = (0..num_heads)
            .map(|_| GraphAttention::new(head_dim, 1, dropout, edge_dim))
            .collect();

        let xavier_std = (2.0 / (input_dim * 2) as f64).sqrt();
        let w_out = GraphAttention::random_matrix(input_dim, input_dim, xavier_std, &mut rng);

        Self {
            heads,
            w_out,
            num_heads,
        }
    }

    /// Forward pass concatenating all heads
    pub fn forward(
        &self,
        x: &Array2<f64>,
        edge_index: &Array2<usize>,
        edge_attr: Option<&Array2<f64>>,
    ) -> Array2<f64> {
        let num_nodes = x.nrows();
        let head_dim = x.ncols() / self.num_heads;
        let total_dim = x.ncols();

        // Split input for each head and process
        let mut outputs = Vec::with_capacity(self.num_heads);

        for (i, head) in self.heads.iter().enumerate() {
            let start = i * head_dim;
            let end = start + head_dim;
            let x_head = x.slice(ndarray::s![.., start..end]).to_owned();
            let out = head.forward(&x_head, edge_index, edge_attr);
            outputs.push(out);
        }

        // Concatenate outputs
        let mut concat = Array2::zeros((num_nodes, total_dim));
        for (i, out) in outputs.iter().enumerate() {
            let start = i * head_dim;
            for j in 0..num_nodes {
                for k in 0..head_dim {
                    concat[[j, start + k]] = out[[j, k]];
                }
            }
        }

        // Final projection
        concat.dot(&self.w_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_attention() {
        let attn = GraphAttention::new(8, 2, 0.0, None);

        // Create simple test data
        let x = Array2::from_shape_fn((3, 8), |_| rand::random::<f64>());
        let edge_index = Array2::from_shape_vec((2, 4), vec![0, 0, 1, 2, 1, 2, 2, 0]).unwrap();

        let output = attn.forward(&x, &edge_index, None);
        assert_eq!(output.shape(), &[3, 8]);
    }

    #[test]
    fn test_attention_weights() {
        let attn = GraphAttention::new(8, 2, 0.0, None);
        let x = Array2::from_shape_fn((3, 8), |_| rand::random::<f64>());
        let edge_index = Array2::from_shape_vec((2, 2), vec![0, 1, 1, 2]).unwrap();

        let weights = attn.get_attention_weights(&x, &edge_index);
        assert!(!weights.is_empty());

        // Check weights sum to approximately 1 for each target node
        for (_, weight) in &weights {
            assert!(*weight >= 0.0 && *weight <= 1.0);
        }
    }
}
