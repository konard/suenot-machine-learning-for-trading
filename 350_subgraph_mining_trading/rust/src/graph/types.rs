//! Core graph types for financial networks.

use petgraph::graph::{DiGraph, NodeIndex, UnGraph};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for a node (asset/symbol)
pub type NodeId = String;

/// Weight of an edge (correlation strength, volume, etc.)
pub type EdgeWeight = f64;

/// Type of graph being constructed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphType {
    /// Undirected correlation network
    Correlation,
    /// Directed lead-lag network
    LeadLag,
    /// Order flow graph
    OrderFlow,
    /// Sector relationship graph
    Sector,
}

/// A node in the financial graph representing an asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Symbol/identifier (e.g., "BTCUSDT")
    pub symbol: String,
    /// Current price
    pub price: f64,
    /// 24h volume
    pub volume: f64,
    /// 24h price change percentage
    pub change_24h: f64,
    /// Market cap (if available)
    pub market_cap: Option<f64>,
    /// Sector/category
    pub sector: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Node {
    /// Create a new node
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            price: 0.0,
            volume: 0.0,
            change_24h: 0.0,
            market_cap: None,
            sector: None,
            metadata: HashMap::new(),
        }
    }

    /// Set price
    pub fn with_price(mut self, price: f64) -> Self {
        self.price = price;
        self
    }

    /// Set volume
    pub fn with_volume(mut self, volume: f64) -> Self {
        self.volume = volume;
        self
    }

    /// Set 24h change
    pub fn with_change(mut self, change: f64) -> Self {
        self.change_24h = change;
        self
    }

    /// Set sector
    pub fn with_sector(mut self, sector: impl Into<String>) -> Self {
        self.sector = Some(sector.into());
        self
    }
}

/// An edge in the financial graph representing a relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Source node symbol
    pub from: String,
    /// Target node symbol
    pub to: String,
    /// Edge weight (correlation, volume, etc.)
    pub weight: EdgeWeight,
    /// Type of relationship
    pub edge_type: EdgeType,
    /// Timestamp when relationship was computed
    pub timestamp: i64,
}

/// Type of edge relationship
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeType {
    /// Positive correlation
    PositiveCorrelation,
    /// Negative correlation
    NegativeCorrelation,
    /// Lead-lag (direction matters)
    LeadLag,
    /// Transaction flow
    Transaction,
    /// Sector membership
    SameSector,
}

impl Edge {
    /// Create a new edge
    pub fn new(from: impl Into<String>, to: impl Into<String>, weight: EdgeWeight) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            weight,
            edge_type: EdgeType::PositiveCorrelation,
            timestamp: chrono::Utc::now().timestamp(),
        }
    }

    /// Set edge type
    pub fn with_type(mut self, edge_type: EdgeType) -> Self {
        self.edge_type = edge_type;
        self
    }
}

/// Main financial graph structure
#[derive(Debug, Clone)]
pub struct FinancialGraph {
    /// Underlying petgraph structure (undirected)
    pub graph: UnGraph<Node, EdgeWeight>,
    /// Directed version for lead-lag analysis
    pub directed_graph: Option<DiGraph<Node, EdgeWeight>>,
    /// Symbol to node index mapping
    pub symbol_index: HashMap<String, NodeIndex>,
    /// Graph type
    pub graph_type: GraphType,
    /// Timestamp of graph construction
    pub timestamp: i64,
    /// Number of time periods used to build graph
    pub window_size: usize,
    /// Correlation/weight threshold used
    pub threshold: f64,
}

impl FinancialGraph {
    /// Create a new empty financial graph
    pub fn new(graph_type: GraphType) -> Self {
        Self {
            graph: UnGraph::new_undirected(),
            directed_graph: None,
            symbol_index: HashMap::new(),
            graph_type,
            timestamp: chrono::Utc::now().timestamp(),
            window_size: 0,
            threshold: 0.0,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: Node) -> NodeIndex {
        let symbol = node.symbol.clone();
        let idx = self.graph.add_node(node);
        self.symbol_index.insert(symbol, idx);
        idx
    }

    /// Add an edge between two nodes
    pub fn add_edge(&mut self, from: &str, to: &str, weight: EdgeWeight) -> Option<petgraph::graph::EdgeIndex> {
        let from_idx = self.symbol_index.get(from)?;
        let to_idx = self.symbol_index.get(to)?;
        Some(self.graph.add_edge(*from_idx, *to_idx, weight))
    }

    /// Get node by symbol
    pub fn get_node(&self, symbol: &str) -> Option<&Node> {
        let idx = self.symbol_index.get(symbol)?;
        self.graph.node_weight(*idx)
    }

    /// Get all node symbols
    pub fn symbols(&self) -> Vec<&String> {
        self.symbol_index.keys().collect()
    }

    /// Number of nodes
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get all edges as (from, to, weight) tuples
    pub fn edges(&self) -> Vec<(String, String, EdgeWeight)> {
        self.graph
            .edge_indices()
            .filter_map(|e| {
                let (a, b) = self.graph.edge_endpoints(e)?;
                let from = self.graph.node_weight(a)?.symbol.clone();
                let to = self.graph.node_weight(b)?.symbol.clone();
                let weight = *self.graph.edge_weight(e)?;
                Some((from, to, weight))
            })
            .collect()
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, symbol: &str) -> Vec<String> {
        self.symbol_index
            .get(symbol)
            .map(|idx| {
                self.graph
                    .neighbors(*idx)
                    .filter_map(|n| self.graph.node_weight(n).map(|node| node.symbol.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check if edge exists between two nodes
    pub fn has_edge(&self, from: &str, to: &str) -> bool {
        if let (Some(from_idx), Some(to_idx)) = (
            self.symbol_index.get(from),
            self.symbol_index.get(to),
        ) {
            self.graph.find_edge(*from_idx, *to_idx).is_some()
        } else {
            false
        }
    }

    /// Get edge weight between two nodes
    pub fn edge_weight(&self, from: &str, to: &str) -> Option<EdgeWeight> {
        let from_idx = self.symbol_index.get(from)?;
        let to_idx = self.symbol_index.get(to)?;
        let edge_idx = self.graph.find_edge(*from_idx, *to_idx)?;
        self.graph.edge_weight(edge_idx).copied()
    }

    /// Get density of the graph (edges / max_possible_edges)
    pub fn density(&self) -> f64 {
        let n = self.node_count() as f64;
        if n < 2.0 {
            return 0.0;
        }
        let max_edges = n * (n - 1.0) / 2.0;
        self.edge_count() as f64 / max_edges
    }

    /// Get average degree
    pub fn average_degree(&self) -> f64 {
        if self.node_count() == 0 {
            return 0.0;
        }
        (2.0 * self.edge_count() as f64) / self.node_count() as f64
    }
}

impl Default for FinancialGraph {
    fn default() -> Self {
        Self::new(GraphType::Correlation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = Node::new("BTCUSDT")
            .with_price(50000.0)
            .with_volume(1000000.0)
            .with_change(5.5);

        assert_eq!(node.symbol, "BTCUSDT");
        assert_eq!(node.price, 50000.0);
        assert_eq!(node.volume, 1000000.0);
        assert_eq!(node.change_24h, 5.5);
    }

    #[test]
    fn test_graph_operations() {
        let mut graph = FinancialGraph::new(GraphType::Correlation);

        graph.add_node(Node::new("BTCUSDT"));
        graph.add_node(Node::new("ETHUSDT"));
        graph.add_node(Node::new("SOLUSDT"));

        graph.add_edge("BTCUSDT", "ETHUSDT", 0.85);
        graph.add_edge("ETHUSDT", "SOLUSDT", 0.75);

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
        assert!(graph.has_edge("BTCUSDT", "ETHUSDT"));
        assert!(!graph.has_edge("BTCUSDT", "SOLUSDT"));

        let neighbors = graph.neighbors("ETHUSDT");
        assert_eq!(neighbors.len(), 2);
    }
}
