//! Dynamic graph operations and temporal graph management

use hashbrown::HashMap;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use super::{Edge, EdgeFeatures, GraphConfig, Node, NodeFeatures, NodeId};

/// A temporal edge with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEdge {
    /// Source node ID
    pub source: NodeId,
    /// Target node ID
    pub target: NodeId,
    /// Event timestamp
    pub timestamp: u64,
    /// Edge features at this time
    pub features: EdgeFeatures,
    /// Event type (e.g., "correlation_change", "trade_flow")
    pub event_type: String,
}

impl TemporalEdge {
    pub fn new(
        source: impl Into<NodeId>,
        target: impl Into<NodeId>,
        timestamp: u64,
        features: EdgeFeatures,
        event_type: impl Into<String>,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            timestamp,
            features,
            event_type: event_type.into(),
        }
    }
}

/// A snapshot of the graph at a specific time
#[derive(Debug, Clone)]
pub struct GraphSnapshot {
    /// Snapshot timestamp
    pub timestamp: u64,
    /// Node embeddings at this time
    pub node_embeddings: HashMap<NodeId, Array1<f64>>,
    /// Adjacency matrix
    pub adjacency: Array2<f64>,
    /// Node order for adjacency matrix
    pub node_order: Vec<NodeId>,
}

/// Dynamic graph that evolves over time
#[derive(Debug, Clone)]
pub struct DynamicGraph {
    /// Configuration
    pub config: GraphConfig,
    /// All nodes in the graph
    nodes: HashMap<NodeId, Node>,
    /// All edges in the graph
    edges: HashMap<(NodeId, NodeId), Edge>,
    /// Temporal edge history
    temporal_history: VecDeque<TemporalEdge>,
    /// Maximum temporal history size
    max_history: usize,
    /// Graph snapshots for temporal analysis
    snapshots: VecDeque<GraphSnapshot>,
    /// Maximum number of snapshots
    max_snapshots: usize,
    /// Current timestamp
    current_time: u64,
}

impl DynamicGraph {
    /// Create a new dynamic graph
    pub fn new() -> Self {
        Self::with_config(GraphConfig::default())
    }

    /// Create a new dynamic graph with configuration
    pub fn with_config(config: GraphConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            edges: HashMap::new(),
            temporal_history: VecDeque::new(),
            max_history: 10000,
            snapshots: VecDeque::new(),
            max_snapshots: 100,
            current_time: 0,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, id: impl Into<NodeId>, features: NodeFeatures) -> bool {
        let id = id.into();
        if self.nodes.len() >= self.config.max_nodes {
            return false;
        }

        let node = Node::new(id.clone(), features);
        self.nodes.insert(id, node);
        true
    }

    /// Remove a node and its edges
    pub fn remove_node(&mut self, id: &str) -> Option<Node> {
        // Remove all edges connected to this node
        self.edges.retain(|(src, tgt), _| src != id && tgt != id);
        self.nodes.remove(id)
    }

    /// Get a node by ID
    pub fn get_node(&self, id: &str) -> Option<&Node> {
        self.nodes.get(id)
    }

    /// Get a mutable node by ID
    pub fn get_node_mut(&mut self, id: &str) -> Option<&mut Node> {
        self.nodes.get_mut(id)
    }

    /// Update node features
    pub fn update_node(&mut self, id: &str, features: NodeFeatures) -> bool {
        if let Some(node) = self.nodes.get_mut(id) {
            node.update_features(features);
            true
        } else {
            false
        }
    }

    /// Add an edge between nodes
    pub fn add_edge(
        &mut self,
        source: impl Into<NodeId>,
        target: impl Into<NodeId>,
        features: EdgeFeatures,
    ) -> bool {
        let source = source.into();
        let target = target.into();

        // Check if both nodes exist
        if !self.nodes.contains_key(&source) || !self.nodes.contains_key(&target) {
            return false;
        }

        let edge = Edge::new(source.clone(), target.clone(), features.clone());
        self.edges.insert((source.clone(), target.clone()), edge);

        // Record temporal edge
        if self.config.temporal_edges {
            let temporal = TemporalEdge::new(
                source,
                target,
                self.current_time,
                features,
                "edge_added",
            );
            self.add_temporal_edge(temporal);
        }

        true
    }

    /// Remove an edge
    pub fn remove_edge(&mut self, source: &str, target: &str) -> Option<Edge> {
        self.edges.remove(&(source.to_string(), target.to_string()))
    }

    /// Get an edge by node IDs
    pub fn get_edge(&self, source: &str, target: &str) -> Option<&Edge> {
        self.edges.get(&(source.to_string(), target.to_string()))
    }

    /// Update an edge
    pub fn update_edge(&mut self, source: &str, target: &str, correlation: f64) -> bool {
        let key = (source.to_string(), target.to_string());
        if let Some(edge) = self.edges.get_mut(&key) {
            edge.update(correlation, self.current_time);

            // Record temporal edge update
            if self.config.temporal_edges {
                let temporal = TemporalEdge::new(
                    source,
                    target,
                    self.current_time,
                    edge.features.clone(),
                    "edge_updated",
                );
                self.add_temporal_edge(temporal);
            }
            true
        } else {
            false
        }
    }

    /// Add a temporal edge to history
    fn add_temporal_edge(&mut self, edge: TemporalEdge) {
        self.temporal_history.push_back(edge);
        if self.temporal_history.len() > self.max_history {
            self.temporal_history.pop_front();
        }
    }

    /// Get neighbors of a node
    pub fn get_neighbors(&self, id: &str) -> Vec<&Node> {
        let mut neighbors = Vec::new();

        for ((src, tgt), edge) in &self.edges {
            if src == id {
                if let Some(node) = self.nodes.get(tgt) {
                    neighbors.push(node);
                }
            } else if edge.bidirectional && tgt == id {
                if let Some(node) = self.nodes.get(src) {
                    neighbors.push(node);
                }
            }
        }

        neighbors
    }

    /// Get edges connected to a node
    pub fn get_node_edges(&self, id: &str) -> Vec<&Edge> {
        self.edges
            .values()
            .filter(|e| e.source == id || (e.bidirectional && e.target == id))
            .collect()
    }

    /// Get all nodes
    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    /// Get all edges
    pub fn edges(&self) -> impl Iterator<Item = &Edge> {
        self.edges.values()
    }

    /// Get number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get number of edges
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get node IDs in consistent order
    pub fn node_ids(&self) -> Vec<NodeId> {
        let mut ids: Vec<_> = self.nodes.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Build adjacency matrix
    pub fn adjacency_matrix(&self) -> (Array2<f64>, Vec<NodeId>) {
        let node_ids = self.node_ids();
        let n = node_ids.len();
        let mut adj = Array2::zeros((n, n));

        let id_to_idx: HashMap<_, _> = node_ids.iter().enumerate().map(|(i, id)| (id, i)).collect();

        for edge in self.edges.values() {
            if let (Some(&i), Some(&j)) = (id_to_idx.get(&edge.source), id_to_idx.get(&edge.target)) {
                adj[[i, j]] = edge.features.weight;
                if edge.bidirectional {
                    adj[[j, i]] = edge.features.weight;
                }
            }
        }

        (adj, node_ids)
    }

    /// Build feature matrix
    pub fn feature_matrix(&self) -> (Array2<f64>, Vec<NodeId>) {
        let node_ids = self.node_ids();
        let n = node_ids.len();
        let d = NodeFeatures::feature_dim();
        let mut features = Array2::zeros((n, d));

        for (i, id) in node_ids.iter().enumerate() {
            if let Some(node) = self.nodes.get(id) {
                let vec = node.get_feature_vector();
                for (j, &val) in vec.iter().enumerate() {
                    features[[i, j]] = val;
                }
            }
        }

        (features, node_ids)
    }

    /// Update graph time
    pub fn tick(&mut self, timestamp: u64) {
        self.current_time = timestamp;
    }

    /// Prune weak edges
    pub fn prune_edges(&mut self, threshold: f64) -> usize {
        let before = self.edges.len();
        self.edges.retain(|_, e| !e.features.should_prune(threshold));
        before - self.edges.len()
    }

    /// Update correlations between all pairs
    pub fn update_correlations(&mut self, returns: &HashMap<NodeId, Vec<f64>>, threshold: f64) {
        let node_ids = self.node_ids();

        for i in 0..node_ids.len() {
            for j in (i + 1)..node_ids.len() {
                let id_i = &node_ids[i];
                let id_j = &node_ids[j];

                if let (Some(ret_i), Some(ret_j)) = (returns.get(id_i), returns.get(id_j)) {
                    let corr = compute_correlation(ret_i, ret_j);

                    if corr.abs() >= threshold {
                        // Add or update edge
                        let key = (id_i.clone(), id_j.clone());
                        if self.edges.contains_key(&key) {
                            self.update_edge(id_i, id_j, corr);
                        } else {
                            let features = EdgeFeatures::with_correlation(corr, self.current_time);
                            self.add_edge(id_i.clone(), id_j.clone(), features);
                        }
                    } else {
                        // Remove weak edge if exists
                        self.remove_edge(id_i, id_j);
                    }
                }
            }
        }
    }

    /// Take a snapshot of current graph state
    pub fn snapshot(&mut self) -> GraphSnapshot {
        let (adjacency, node_order) = self.adjacency_matrix();
        let node_embeddings: HashMap<_, _> = self
            .nodes
            .iter()
            .filter_map(|(id, node)| node.get_embedding().map(|e| (id.clone(), e.clone())))
            .collect();

        let snapshot = GraphSnapshot {
            timestamp: self.current_time,
            node_embeddings,
            adjacency,
            node_order,
        };

        self.snapshots.push_back(snapshot.clone());
        if self.snapshots.len() > self.max_snapshots {
            self.snapshots.pop_front();
        }

        snapshot
    }

    /// Get temporal edges in time range
    pub fn temporal_edges_in_range(&self, start: u64, end: u64) -> Vec<&TemporalEdge> {
        self.temporal_history
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .collect()
    }

    /// Get recent temporal edges
    pub fn recent_temporal_edges(&self, count: usize) -> Vec<&TemporalEdge> {
        self.temporal_history.iter().rev().take(count).collect()
    }

    /// Get graph statistics
    pub fn stats(&self) -> GraphStats {
        let (adj, _) = self.adjacency_matrix();
        let edge_weights: Vec<f64> = self.edges.values().map(|e| e.features.weight).collect();

        GraphStats {
            node_count: self.node_count(),
            edge_count: self.edge_count(),
            density: if self.node_count() > 1 {
                2.0 * self.edge_count() as f64
                    / (self.node_count() as f64 * (self.node_count() - 1) as f64)
            } else {
                0.0
            },
            avg_degree: if self.node_count() > 0 {
                2.0 * self.edge_count() as f64 / self.node_count() as f64
            } else {
                0.0
            },
            avg_edge_weight: if !edge_weights.is_empty() {
                edge_weights.iter().sum::<f64>() / edge_weights.len() as f64
            } else {
                0.0
            },
            spectral_norm: spectral_norm(&adj),
        }
    }
}

impl Default for DynamicGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph statistics
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub density: f64,
    pub avg_degree: f64,
    pub avg_edge_weight: f64,
    pub spectral_norm: f64,
}

/// Compute Pearson correlation between two vectors
fn compute_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

/// Compute spectral norm (largest singular value) approximation
fn spectral_norm(matrix: &Array2<f64>) -> f64 {
    // Power iteration for largest eigenvalue approximation
    let n = matrix.nrows();
    if n == 0 {
        return 0.0;
    }

    let mut v = Array1::ones(n) / (n as f64).sqrt();

    for _ in 0..20 {
        let av = matrix.dot(&v);
        let norm = av.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            return 0.0;
        }
        v = av / norm;
    }

    let av = matrix.dot(&v);
    av.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_graph_creation() {
        let graph = DynamicGraph::new();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_add_nodes_and_edges() {
        let mut graph = DynamicGraph::new();

        graph.add_node("BTC", NodeFeatures::new(50000.0, 1_000_000.0, 1000));
        graph.add_node("ETH", NodeFeatures::new(3000.0, 500_000.0, 1000));

        assert_eq!(graph.node_count(), 2);

        let features = EdgeFeatures::with_correlation(0.85, 1000);
        graph.add_edge("BTC", "ETH", features);

        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_neighbors() {
        let mut graph = DynamicGraph::new();

        graph.add_node("A", NodeFeatures::default());
        graph.add_node("B", NodeFeatures::default());
        graph.add_node("C", NodeFeatures::default());

        graph.add_edge("A", "B", EdgeFeatures::default());
        graph.add_edge("A", "C", EdgeFeatures::default());

        let neighbors = graph.get_neighbors("A");
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_adjacency_matrix() {
        let mut graph = DynamicGraph::new();

        graph.add_node("A", NodeFeatures::default());
        graph.add_node("B", NodeFeatures::default());

        let mut features = EdgeFeatures::default();
        features.weight = 0.8;
        graph.add_edge("A", "B", features);

        let (adj, _) = graph.adjacency_matrix();
        assert!(adj[[0, 1]] > 0.0 || adj[[1, 0]] > 0.0);
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let corr = compute_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 0.001);

        let z = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr_neg = compute_correlation(&x, &z);
        assert!((corr_neg + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_graph_stats() {
        let mut graph = DynamicGraph::new();

        graph.add_node("A", NodeFeatures::default());
        graph.add_node("B", NodeFeatures::default());
        graph.add_node("C", NodeFeatures::default());

        graph.add_edge("A", "B", EdgeFeatures::default());
        graph.add_edge("B", "C", EdgeFeatures::default());

        let stats = graph.stats();
        assert_eq!(stats.node_count, 3);
        assert_eq!(stats.edge_count, 2);
    }
}
