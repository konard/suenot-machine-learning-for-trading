//! Graph data structures and utilities for market graph construction.
//!
//! This module provides the core graph data structures used to represent
//! market relationships and utilities for building graphs from market data.

mod construction;
mod features;

pub use construction::*;
pub use features::*;

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a node (asset) in the market graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique identifier for the node
    pub id: usize,
    /// Symbol of the asset (e.g., "BTCUSDT")
    pub symbol: String,
    /// Feature vector for this node
    pub features: Array1<f64>,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl Node {
    /// Create a new node with the given symbol and features.
    pub fn new(id: usize, symbol: impl Into<String>, features: Array1<f64>) -> Self {
        Self {
            id,
            symbol: symbol.into(),
            features,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to this node.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get the feature dimension.
    pub fn feature_dim(&self) -> usize {
        self.features.len()
    }
}

/// Represents an edge (relationship) in the market graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Source node ID
    pub source: usize,
    /// Target node ID
    pub target: usize,
    /// Edge weight (e.g., correlation strength)
    pub weight: f64,
    /// Optional edge features
    pub features: Option<Array1<f64>>,
    /// Edge type for heterogeneous graphs
    pub edge_type: EdgeType,
}

impl Edge {
    /// Create a new edge between two nodes.
    pub fn new(source: usize, target: usize, weight: f64) -> Self {
        Self {
            source,
            target,
            weight,
            features: None,
            edge_type: EdgeType::Correlation,
        }
    }

    /// Set the edge type.
    pub fn with_type(mut self, edge_type: EdgeType) -> Self {
        self.edge_type = edge_type;
        self
    }

    /// Set edge features.
    pub fn with_features(mut self, features: Array1<f64>) -> Self {
        self.features = Some(features);
        self
    }
}

/// Types of edges in the market graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    /// Based on return correlation
    Correlation,
    /// Same sector/category
    Sector,
    /// Lead-lag relationship
    LeadLag,
    /// Liquidity/trading pair connection
    Liquidity,
    /// Protocol dependency (e.g., tokens on same blockchain)
    Protocol,
}

/// A market graph representing relationships between assets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketGraph {
    /// Nodes in the graph
    pub nodes: Vec<Node>,
    /// Edges in the graph
    pub edges: Vec<Edge>,
    /// Symbol to node ID mapping
    pub symbol_to_id: HashMap<String, usize>,
    /// Adjacency matrix (sparse representation would be better for large graphs)
    adjacency: Option<Array2<f64>>,
}

impl MarketGraph {
    /// Create a new empty market graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            symbol_to_id: HashMap::new(),
            adjacency: None,
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, symbol: impl Into<String>, features: Array1<f64>) -> usize {
        let symbol = symbol.into();
        let id = self.nodes.len();
        self.symbol_to_id.insert(symbol.clone(), id);
        self.nodes.push(Node::new(id, symbol, features));
        self.adjacency = None; // Invalidate cached adjacency matrix
        id
    }

    /// Add an edge between two nodes.
    pub fn add_edge(&mut self, source: usize, target: usize, weight: f64) -> &mut Edge {
        let edge = Edge::new(source, target, weight);
        self.edges.push(edge);
        self.adjacency = None; // Invalidate cached adjacency matrix
        self.edges.last_mut().unwrap()
    }

    /// Add an edge by symbol names.
    pub fn add_edge_by_symbol(
        &mut self,
        source_symbol: &str,
        target_symbol: &str,
        weight: f64,
    ) -> Option<&mut Edge> {
        let source = *self.symbol_to_id.get(source_symbol)?;
        let target = *self.symbol_to_id.get(target_symbol)?;
        Some(self.add_edge(source, target, weight))
    }

    /// Get the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: usize) -> Option<&Node> {
        self.nodes.get(id)
    }

    /// Get a node by symbol.
    pub fn get_node_by_symbol(&self, symbol: &str) -> Option<&Node> {
        self.symbol_to_id
            .get(symbol)
            .and_then(|&id| self.nodes.get(id))
    }

    /// Get all neighbors of a node.
    pub fn neighbors(&self, node_id: usize) -> Vec<(usize, f64)> {
        self.edges
            .iter()
            .filter_map(|e| {
                if e.source == node_id {
                    Some((e.target, e.weight))
                } else if e.target == node_id {
                    Some((e.source, e.weight))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get the adjacency matrix, computing it if necessary.
    pub fn adjacency_matrix(&mut self) -> &Array2<f64> {
        if self.adjacency.is_none() {
            self.compute_adjacency();
        }
        self.adjacency.as_ref().unwrap()
    }

    /// Compute the adjacency matrix from edges.
    fn compute_adjacency(&mut self) {
        let n = self.nodes.len();
        let mut adj = Array2::zeros((n, n));

        for edge in &self.edges {
            adj[[edge.source, edge.target]] = edge.weight;
            adj[[edge.target, edge.source]] = edge.weight; // Assuming undirected graph
        }

        // Add self-loops
        for i in 0..n {
            adj[[i, i]] = 1.0;
        }

        self.adjacency = Some(adj);
    }

    /// Get the feature matrix (nodes x features).
    pub fn feature_matrix(&self) -> Array2<f64> {
        if self.nodes.is_empty() {
            return Array2::zeros((0, 0));
        }

        let n = self.nodes.len();
        let d = self.nodes[0].feature_dim();
        let mut features = Array2::zeros((n, d));

        for (i, node) in self.nodes.iter().enumerate() {
            features.row_mut(i).assign(&node.features);
        }

        features
    }

    /// Get the normalized adjacency matrix (D^-1/2 * A * D^-1/2).
    pub fn normalized_adjacency(&mut self) -> Array2<f64> {
        let adj = self.adjacency_matrix().clone();
        let n = adj.nrows();

        // Compute degree matrix
        let degrees: Vec<f64> = (0..n)
            .map(|i| adj.row(i).sum().max(1e-10))
            .collect();

        // Compute D^-1/2
        let d_inv_sqrt: Vec<f64> = degrees.iter().map(|d| 1.0 / d.sqrt()).collect();

        // Compute D^-1/2 * A * D^-1/2
        let mut normalized = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                normalized[[i, j]] = d_inv_sqrt[i] * adj[[i, j]] * d_inv_sqrt[j];
            }
        }

        normalized
    }

    /// Update node features.
    pub fn update_node_features(&mut self, node_id: usize, features: Array1<f64>) {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.features = features;
        }
    }

    /// Get all symbols in the graph.
    pub fn symbols(&self) -> Vec<&str> {
        self.nodes.iter().map(|n| n.symbol.as_str()).collect()
    }
}

impl Default for MarketGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_graph_creation() {
        let mut graph = MarketGraph::new();

        let btc_features = array![1.0, 0.5, 0.3];
        let eth_features = array![0.8, 0.6, 0.4];

        let btc_id = graph.add_node("BTCUSDT", btc_features);
        let eth_id = graph.add_node("ETHUSDT", eth_features);

        graph.add_edge(btc_id, eth_id, 0.9);

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_neighbors() {
        let mut graph = MarketGraph::new();

        let features = array![1.0, 0.5];
        let a = graph.add_node("A", features.clone());
        let b = graph.add_node("B", features.clone());
        let c = graph.add_node("C", features);

        graph.add_edge(a, b, 0.8);
        graph.add_edge(a, c, 0.6);

        let neighbors = graph.neighbors(a);
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_adjacency_matrix() {
        let mut graph = MarketGraph::new();

        let features = array![1.0];
        graph.add_node("A", features.clone());
        graph.add_node("B", features.clone());
        graph.add_node("C", features);

        graph.add_edge(0, 1, 0.5);
        graph.add_edge(1, 2, 0.3);

        let adj = graph.adjacency_matrix();
        assert_eq!(adj[[0, 1]], 0.5);
        assert_eq!(adj[[1, 0]], 0.5);
        assert_eq!(adj[[1, 2]], 0.3);
        assert_eq!(adj[[0, 0]], 1.0); // Self-loop
    }
}
