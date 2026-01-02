//! Graph construction and analysis module.
//!
//! This module provides various methods for constructing financial graphs:
//! - Correlation networks
//! - Visibility graphs
//! - K-nearest neighbor graphs
//! - Minimum spanning trees

mod builder;
mod correlation;
mod metrics;
mod visibility;

pub use builder::{CorrelationMethod, GraphBuilder};
pub use correlation::{pearson_correlation, spearman_correlation, CorrelationMatrix};
pub use metrics::GraphMetrics;
pub use visibility::VisibilityGraph;

use petgraph::graph::{NodeIndex, UnGraph};
use std::collections::HashMap;

/// A market graph representing relationships between assets
#[derive(Debug, Clone)]
pub struct MarketGraph {
    /// The underlying petgraph structure
    pub graph: UnGraph<String, f64>,
    /// Mapping from symbol name to node index
    pub symbol_to_node: HashMap<String, NodeIndex>,
    /// Mapping from node index to symbol name
    pub node_to_symbol: HashMap<NodeIndex, String>,
    /// Graph construction parameters
    pub params: GraphParams,
}

/// Parameters used to construct the graph
#[derive(Debug, Clone)]
pub struct GraphParams {
    /// Method used for correlation calculation
    pub correlation_method: CorrelationMethod,
    /// Threshold for edge inclusion (if applicable)
    pub threshold: Option<f64>,
    /// K for KNN (if applicable)
    pub k: Option<usize>,
    /// Window size for rolling calculations
    pub window: Option<usize>,
}

impl Default for GraphParams {
    fn default() -> Self {
        Self {
            correlation_method: CorrelationMethod::Pearson,
            threshold: Some(0.5),
            k: None,
            window: None,
        }
    }
}

impl MarketGraph {
    /// Create a new empty market graph
    pub fn new() -> Self {
        Self {
            graph: UnGraph::new_undirected(),
            symbol_to_node: HashMap::new(),
            node_to_symbol: HashMap::new(),
            params: GraphParams::default(),
        }
    }

    /// Create a market graph with specified symbols
    pub fn with_symbols(symbols: &[String]) -> Self {
        let mut graph = UnGraph::new_undirected();
        let mut symbol_to_node = HashMap::new();
        let mut node_to_symbol = HashMap::new();

        for symbol in symbols {
            let idx = graph.add_node(symbol.clone());
            symbol_to_node.insert(symbol.clone(), idx);
            node_to_symbol.insert(idx, symbol.clone());
        }

        Self {
            graph,
            symbol_to_node,
            node_to_symbol,
            params: GraphParams::default(),
        }
    }

    /// Add a node (symbol) to the graph
    pub fn add_node(&mut self, symbol: &str) -> NodeIndex {
        if let Some(&idx) = self.symbol_to_node.get(symbol) {
            return idx;
        }

        let idx = self.graph.add_node(symbol.to_string());
        self.symbol_to_node.insert(symbol.to_string(), idx);
        self.node_to_symbol.insert(idx, symbol.to_string());
        idx
    }

    /// Add an edge between two symbols
    pub fn add_edge(&mut self, symbol1: &str, symbol2: &str, weight: f64) {
        let idx1 = self.add_node(symbol1);
        let idx2 = self.add_node(symbol2);

        // Remove existing edge if any
        if let Some(edge) = self.graph.find_edge(idx1, idx2) {
            self.graph.remove_edge(edge);
        }

        self.graph.add_edge(idx1, idx2, weight);
    }

    /// Get the number of nodes
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of edges
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Calculate graph density
    pub fn density(&self) -> f64 {
        let n = self.node_count() as f64;
        if n <= 1.0 {
            return 0.0;
        }

        let max_edges = n * (n - 1.0) / 2.0;
        self.edge_count() as f64 / max_edges
    }

    /// Get all neighbors of a node
    pub fn neighbors(&self, symbol: &str) -> Vec<String> {
        if let Some(&idx) = self.symbol_to_node.get(symbol) {
            self.graph
                .neighbors(idx)
                .filter_map(|neighbor| self.node_to_symbol.get(&neighbor).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get edge weight between two symbols
    pub fn edge_weight(&self, symbol1: &str, symbol2: &str) -> Option<f64> {
        let idx1 = self.symbol_to_node.get(symbol1)?;
        let idx2 = self.symbol_to_node.get(symbol2)?;

        self.graph
            .find_edge(*idx1, *idx2)
            .and_then(|edge| self.graph.edge_weight(edge).copied())
    }

    /// Get degree of a node
    pub fn degree(&self, symbol: &str) -> usize {
        if let Some(&idx) = self.symbol_to_node.get(symbol) {
            self.graph.neighbors(idx).count()
        } else {
            0
        }
    }

    /// Get all symbols in the graph
    pub fn symbols(&self) -> Vec<String> {
        self.symbol_to_node.keys().cloned().collect()
    }

    /// Get edges as (symbol1, symbol2, weight) tuples
    pub fn edges(&self) -> Vec<(String, String, f64)> {
        self.graph
            .edge_indices()
            .filter_map(|edge| {
                let (a, b) = self.graph.edge_endpoints(edge)?;
                let weight = self.graph.edge_weight(edge)?;
                let sym_a = self.node_to_symbol.get(&a)?;
                let sym_b = self.node_to_symbol.get(&b)?;
                Some((sym_a.clone(), sym_b.clone(), *weight))
            })
            .collect()
    }

    /// Get the adjacency matrix
    pub fn adjacency_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.node_count();
        let mut matrix = vec![vec![0.0; n]; n];

        let symbols: Vec<_> = self.symbols();
        let sym_to_idx: HashMap<_, _> = symbols
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        for (s1, s2, weight) in self.edges() {
            if let (Some(&i), Some(&j)) = (sym_to_idx.get(&s1), sym_to_idx.get(&s2)) {
                matrix[i][j] = weight;
                matrix[j][i] = weight;
            }
        }

        matrix
    }

    /// Filter edges by threshold
    pub fn filter_by_threshold(&mut self, threshold: f64) {
        let edges_to_remove: Vec<_> = self
            .graph
            .edge_indices()
            .filter(|&edge| {
                self.graph
                    .edge_weight(edge)
                    .map(|w| w.abs() < threshold)
                    .unwrap_or(true)
            })
            .collect();

        for edge in edges_to_remove {
            self.graph.remove_edge(edge);
        }
    }

    /// Get subgraph containing only specified symbols
    pub fn subgraph(&self, symbols: &[String]) -> MarketGraph {
        let mut subgraph = MarketGraph::new();
        subgraph.params = self.params.clone();

        // Add nodes
        for symbol in symbols {
            if self.symbol_to_node.contains_key(symbol) {
                subgraph.add_node(symbol);
            }
        }

        // Add edges
        for (s1, s2, weight) in self.edges() {
            if symbols.contains(&s1) && symbols.contains(&s2) {
                subgraph.add_edge(&s1, &s2, weight);
            }
        }

        subgraph
    }

    /// Calculate average edge weight
    pub fn average_edge_weight(&self) -> f64 {
        let edges: Vec<f64> = self
            .graph
            .edge_weights()
            .copied()
            .collect();

        if edges.is_empty() {
            return 0.0;
        }

        edges.iter().sum::<f64>() / edges.len() as f64
    }

    /// Get strongly connected components (for weighted graph, based on threshold)
    pub fn connected_components(&self) -> Vec<Vec<String>> {
        use petgraph::algo::connected_components;

        let num_components = connected_components(&self.graph);
        let mut components: Vec<Vec<String>> = vec![Vec::new(); num_components];

        // Get component ID for each node
        for (symbol, &idx) in &self.symbol_to_node {
            // Simple DFS to find component
            let mut visited = vec![false; self.graph.node_count()];
            let component_id = self.find_component_id(idx, &mut visited);
            if component_id < components.len() {
                components[component_id].push(symbol.clone());
            }
        }

        components.retain(|c| !c.is_empty());
        components
    }

    fn find_component_id(&self, start: NodeIndex, visited: &mut [bool]) -> usize {
        let start_idx = start.index();
        if start_idx >= visited.len() {
            return 0;
        }

        // Simple traversal to identify component
        let mut min_idx = start_idx;
        let mut stack = vec![start];

        while let Some(node) = stack.pop() {
            let idx = node.index();
            if idx >= visited.len() || visited[idx] {
                continue;
            }
            visited[idx] = true;
            min_idx = min_idx.min(idx);

            for neighbor in self.graph.neighbors(node) {
                if neighbor.index() < visited.len() && !visited[neighbor.index()] {
                    stack.push(neighbor);
                }
            }
        }

        min_idx
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

    #[test]
    fn test_market_graph_creation() {
        let symbols = vec!["BTC".to_string(), "ETH".to_string(), "SOL".to_string()];
        let graph = MarketGraph::with_symbols(&symbols);

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_add_edges() {
        let mut graph = MarketGraph::new();

        graph.add_edge("BTC", "ETH", 0.85);
        graph.add_edge("BTC", "SOL", 0.72);
        graph.add_edge("ETH", "SOL", 0.78);

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 3);
        assert_eq!(graph.edge_weight("BTC", "ETH"), Some(0.85));
    }

    #[test]
    fn test_density() {
        let mut graph = MarketGraph::new();

        graph.add_edge("BTC", "ETH", 0.85);
        graph.add_edge("BTC", "SOL", 0.72);
        graph.add_edge("ETH", "SOL", 0.78);

        // 3 nodes, 3 edges, max edges = 3
        assert!((graph.density() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_neighbors() {
        let mut graph = MarketGraph::new();

        graph.add_edge("BTC", "ETH", 0.85);
        graph.add_edge("BTC", "SOL", 0.72);

        let neighbors = graph.neighbors("BTC");
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&"ETH".to_string()));
        assert!(neighbors.contains(&"SOL".to_string()));
    }

    #[test]
    fn test_filter_by_threshold() {
        let mut graph = MarketGraph::new();

        graph.add_edge("BTC", "ETH", 0.85);
        graph.add_edge("BTC", "SOL", 0.45);
        graph.add_edge("ETH", "SOL", 0.78);

        graph.filter_by_threshold(0.5);

        assert_eq!(graph.edge_count(), 2);
        assert!(graph.edge_weight("BTC", "SOL").is_none());
    }
}
