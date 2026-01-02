//! Market graph data structure
//!
//! Represents the cryptocurrency market as a graph with nodes (assets) and edges (relationships).

use super::edge::{EdgeList, EdgeType, MarketEdge};
use super::node::CryptoNode;
use ndarray::{Array1, Array2};
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;

/// Market graph representation
#[derive(Debug, Clone)]
pub struct MarketGraph {
    /// Nodes (cryptocurrency assets)
    pub nodes: Vec<CryptoNode>,
    /// Edges (relationships between assets)
    pub edges: Vec<MarketEdge>,
    /// Symbol to node index mapping
    symbol_to_idx: HashMap<String, usize>,
    /// Adjacency list for efficient neighbor lookup
    adjacency: HashMap<usize, Vec<usize>>,
    /// Edge index for (source, target) lookup
    edge_index: HashMap<(usize, usize), usize>,
}

impl MarketGraph {
    /// Create a new empty market graph
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            symbol_to_idx: HashMap::new(),
            adjacency: HashMap::new(),
            edge_index: HashMap::new(),
        }
    }

    /// Create from nodes and edges
    pub fn from_nodes_edges(nodes: Vec<CryptoNode>, edges: Vec<MarketEdge>) -> Self {
        let symbol_to_idx: HashMap<String, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.symbol.clone(), i))
            .collect();

        let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut edge_index: HashMap<(usize, usize), usize> = HashMap::new();

        for (i, edge) in edges.iter().enumerate() {
            adjacency.entry(edge.source).or_default().push(edge.target);
            adjacency.entry(edge.target).or_default().push(edge.source);
            edge_index.insert((edge.source, edge.target), i);
            edge_index.insert((edge.target, edge.source), i);
        }

        Self {
            nodes,
            edges,
            symbol_to_idx,
            adjacency,
            edge_index,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: CryptoNode) -> usize {
        let idx = self.nodes.len();
        self.symbol_to_idx.insert(node.symbol.clone(), idx);
        self.nodes.push(node);
        self.adjacency.insert(idx, Vec::new());
        idx
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: MarketEdge) -> usize {
        let idx = self.edges.len();
        self.adjacency.entry(edge.source).or_default().push(edge.target);
        self.adjacency.entry(edge.target).or_default().push(edge.source);
        self.edge_index.insert((edge.source, edge.target), idx);
        self.edge_index.insert((edge.target, edge.source), idx);
        self.edges.push(edge);
        idx
    }

    /// Get node by symbol
    pub fn get_node_by_symbol(&self, symbol: &str) -> Option<&CryptoNode> {
        self.symbol_to_idx.get(symbol).map(|&idx| &self.nodes[idx])
    }

    /// Get node index by symbol
    pub fn get_node_idx(&self, symbol: &str) -> Option<usize> {
        self.symbol_to_idx.get(symbol).copied()
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node_idx: usize) -> &[usize] {
        self.adjacency.get(&node_idx).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get edge between two nodes
    pub fn get_edge(&self, source: usize, target: usize) -> Option<&MarketEdge> {
        self.edge_index
            .get(&(source, target))
            .map(|&idx| &self.edges[idx])
    }

    /// Number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get node degree
    pub fn degree(&self, node_idx: usize) -> usize {
        self.adjacency.get(&node_idx).map(|v| v.len()).unwrap_or(0)
    }

    /// Get average degree
    pub fn avg_degree(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let total_degree: usize = self.nodes.iter().map(|n| self.degree(n.id)).sum();
        total_degree as f64 / self.nodes.len() as f64
    }

    /// Get node feature matrix [num_nodes, feature_dim]
    pub fn node_feature_matrix(&self) -> Option<Array2<f64>> {
        let first_dim = self.nodes.first()?.feature_dim()?;
        let mut matrix = Array2::zeros((self.nodes.len(), first_dim));

        for (i, node) in self.nodes.iter().enumerate() {
            if let Some(features) = &node.features {
                matrix.row_mut(i).assign(features);
            }
        }

        Some(matrix)
    }

    /// Get edge index in COO format [2, num_edges]
    pub fn edge_index_coo(&self) -> Array2<usize> {
        let n = self.edges.len();
        let mut result = Array2::zeros((2, n));

        for (i, edge) in self.edges.iter().enumerate() {
            result[[0, i]] = edge.source;
            result[[1, i]] = edge.target;
        }

        result
    }

    /// Get edge weights as array
    pub fn edge_weights(&self) -> Array1<f64> {
        Array1::from_iter(self.edges.iter().map(|e| e.weight))
    }

    /// Get adjacency matrix
    pub fn adjacency_matrix(&self) -> Array2<f64> {
        let n = self.nodes.len();
        let mut adj = Array2::zeros((n, n));

        for edge in &self.edges {
            adj[[edge.source, edge.target]] = edge.weight;
            adj[[edge.target, edge.source]] = edge.weight;  // Undirected
        }

        adj
    }

    /// Get degree matrix (diagonal)
    pub fn degree_matrix(&self) -> Array2<f64> {
        let n = self.nodes.len();
        let mut deg = Array2::zeros((n, n));

        for i in 0..n {
            deg[[i, i]] = self.degree(i) as f64;
        }

        deg
    }

    /// Get graph Laplacian (L = D - A)
    pub fn laplacian_matrix(&self) -> Array2<f64> {
        let d = self.degree_matrix();
        let a = self.adjacency_matrix();
        d - a
    }

    /// Get normalized adjacency matrix (D^{-1/2} A D^{-1/2})
    pub fn normalized_adjacency(&self) -> Array2<f64> {
        let n = self.nodes.len();
        let adj = self.adjacency_matrix();
        let mut norm_adj = Array2::zeros((n, n));

        // Compute D^{-1/2}
        let degrees: Vec<f64> = (0..n).map(|i| self.degree(i) as f64).collect();

        for i in 0..n {
            for j in 0..n {
                if adj[[i, j]] > 0.0 && degrees[i] > 0.0 && degrees[j] > 0.0 {
                    norm_adj[[i, j]] = adj[[i, j]] / (degrees[i] * degrees[j]).sqrt();
                }
            }
        }

        norm_adj
    }

    /// Convert to EdgeList format
    pub fn to_edge_list(&self) -> EdgeList {
        EdgeList::from_edges(&self.edges)
    }

    /// Get subgraph containing only specified nodes
    pub fn subgraph(&self, node_indices: &[usize]) -> MarketGraph {
        let idx_set: std::collections::HashSet<_> = node_indices.iter().copied().collect();
        let old_to_new: HashMap<usize, usize> = node_indices
            .iter()
            .enumerate()
            .map(|(new, &old)| (old, new))
            .collect();

        let new_nodes: Vec<CryptoNode> = node_indices
            .iter()
            .map(|&i| {
                let mut node = self.nodes[i].clone();
                node.id = old_to_new[&i];
                node
            })
            .collect();

        let new_edges: Vec<MarketEdge> = self
            .edges
            .iter()
            .filter(|e| idx_set.contains(&e.source) && idx_set.contains(&e.target))
            .map(|e| MarketEdge {
                source: old_to_new[&e.source],
                target: old_to_new[&e.target],
                edge_type: e.edge_type,
                weight: e.weight,
                features: e.features.clone(),
            })
            .collect();

        MarketGraph::from_nodes_edges(new_nodes, new_edges)
    }

    /// Filter edges by weight threshold
    pub fn filter_edges_by_weight(&mut self, min_weight: f64) {
        self.edges.retain(|e| e.weight >= min_weight);
        self.rebuild_indices();
    }

    /// Filter edges by type
    pub fn filter_edges_by_type(&mut self, edge_types: &[EdgeType]) {
        self.edges.retain(|e| edge_types.contains(&e.edge_type));
        self.rebuild_indices();
    }

    /// Rebuild adjacency and edge indices
    fn rebuild_indices(&mut self) {
        self.adjacency.clear();
        self.edge_index.clear();

        for (i, edge) in self.edges.iter().enumerate() {
            self.adjacency.entry(edge.source).or_default().push(edge.target);
            self.adjacency.entry(edge.target).or_default().push(edge.source);
            self.edge_index.insert((edge.source, edge.target), i);
            self.edge_index.insert((edge.target, edge.source), i);
        }
    }

    /// Get graph statistics
    pub fn stats(&self) -> GraphStats {
        let num_nodes = self.num_nodes();
        let num_edges = self.num_edges();
        let avg_degree = self.avg_degree();
        let density = if num_nodes > 1 {
            2.0 * num_edges as f64 / (num_nodes * (num_nodes - 1)) as f64
        } else {
            0.0
        };

        GraphStats {
            num_nodes,
            num_edges,
            avg_degree,
            density,
        }
    }
}

impl Default for MarketGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph statistics
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub avg_degree: f64,
    pub density: f64,
}

impl std::fmt::Display for GraphStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Nodes: {}, Edges: {}, Avg Degree: {:.2}, Density: {:.4}",
            self.num_nodes, self.num_edges, self.avg_degree, self.density
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> MarketGraph {
        let nodes = vec![
            CryptoNode::new(0, "BTCUSDT", 45000.0),
            CryptoNode::new(1, "ETHUSDT", 2500.0),
            CryptoNode::new(2, "SOLUSDT", 100.0),
        ];

        let edges = vec![
            MarketEdge::correlation(0, 1, 0.85),
            MarketEdge::correlation(0, 2, 0.70),
            MarketEdge::correlation(1, 2, 0.75),
        ];

        MarketGraph::from_nodes_edges(nodes, edges)
    }

    #[test]
    fn test_graph_creation() {
        let graph = create_test_graph();
        assert_eq!(graph.num_nodes(), 3);
        assert_eq!(graph.num_edges(), 3);
    }

    #[test]
    fn test_neighbors() {
        let graph = create_test_graph();
        let neighbors = graph.neighbors(0);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2));
    }

    #[test]
    fn test_adjacency_matrix() {
        let graph = create_test_graph();
        let adj = graph.adjacency_matrix();
        assert_eq!(adj.shape(), &[3, 3]);
        assert!(adj[[0, 1]] > 0.0);
    }

    #[test]
    fn test_graph_stats() {
        let graph = create_test_graph();
        let stats = graph.stats();
        assert_eq!(stats.num_nodes, 3);
        assert_eq!(stats.num_edges, 3);
        assert_eq!(stats.density, 1.0);  // Complete graph
    }
}
