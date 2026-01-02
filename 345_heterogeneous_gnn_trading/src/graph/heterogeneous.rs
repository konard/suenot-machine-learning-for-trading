//! Heterogeneous graph implementation
//!
//! Main graph data structure supporting multiple node and edge types.

use std::collections::{HashMap, HashSet};
use ndarray::{Array1, Array2};
use hashbrown::HashMap as FastHashMap;

use super::{
    Node, NodeType, NodeId, NodeFeatures,
    Edge, EdgeType, EdgeFeatures,
    GraphSchema, GraphConfig, Metapath, MetapathInstance,
};

/// Heterogeneous graph with typed nodes and edges
#[derive(Debug, Clone)]
pub struct HeterogeneousGraph {
    /// Graph schema
    schema: GraphSchema,
    /// Graph configuration
    config: GraphConfig,
    /// All nodes indexed by ID
    nodes: FastHashMap<NodeId, Node>,
    /// Nodes grouped by type
    nodes_by_type: HashMap<NodeType, HashSet<NodeId>>,
    /// All edges
    edges: Vec<Edge>,
    /// Adjacency list: node_id -> [(neighbor_id, edge_index)]
    adjacency: FastHashMap<NodeId, Vec<(NodeId, usize)>>,
    /// Edges grouped by type
    edges_by_type: HashMap<EdgeType, Vec<usize>>,
    /// Current timestamp
    current_time: u64,
}

impl HeterogeneousGraph {
    /// Create a new heterogeneous graph with schema
    pub fn new(schema: GraphSchema) -> Self {
        Self::with_config(schema, GraphConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(schema: GraphSchema, config: GraphConfig) -> Self {
        let mut nodes_by_type = HashMap::new();
        let mut edges_by_type = HashMap::new();

        for node_type in NodeType::all() {
            nodes_by_type.insert(node_type, HashSet::new());
        }

        for edge_type in EdgeType::all() {
            edges_by_type.insert(edge_type, Vec::new());
        }

        Self {
            schema,
            config,
            nodes: FastHashMap::new(),
            nodes_by_type,
            edges: Vec::new(),
            adjacency: FastHashMap::new(),
            edges_by_type,
            current_time: 0,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, id: impl Into<String>, node_type: NodeType, features: NodeFeatures) {
        let id = id.into();
        let node = Node::new(id.clone(), node_type, features);

        self.nodes.insert(id.clone(), node);
        self.nodes_by_type.get_mut(&node_type).unwrap().insert(id.clone());
        self.adjacency.entry(id).or_insert_with(Vec::new);
    }

    /// Update node features
    pub fn update_node(&mut self, id: &str, features: NodeFeatures) {
        if let Some(node) = self.nodes.get_mut(id) {
            node.update_features(features);
        }
    }

    /// Get a node by ID
    pub fn get_node(&self, id: &str) -> Option<&Node> {
        self.nodes.get(id)
    }

    /// Get all nodes of a specific type
    pub fn get_nodes_by_type(&self, node_type: NodeType) -> Vec<&Node> {
        self.nodes_by_type
            .get(&node_type)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.nodes.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Add an edge to the graph
    pub fn add_edge(
        &mut self,
        source: impl Into<String>,
        target: impl Into<String>,
        edge_type: EdgeType,
        features: EdgeFeatures,
    ) {
        let source = source.into();
        let target = target.into();

        // Validate that nodes exist
        if !self.nodes.contains_key(&source) || !self.nodes.contains_key(&target) {
            return;
        }

        // Validate relation is allowed by schema
        let source_type = self.nodes.get(&source).unwrap().node_type;
        let target_type = self.nodes.get(&target).unwrap().node_type;

        if !self.schema.is_valid_relation(source_type, edge_type, target_type) {
            return;
        }

        let edge = Edge::new(source.clone(), target.clone(), edge_type, features);
        let edge_idx = self.edges.len();

        self.edges.push(edge);
        self.edges_by_type.get_mut(&edge_type).unwrap().push(edge_idx);

        // Update adjacency list
        self.adjacency.get_mut(&source).unwrap().push((target.clone(), edge_idx));

        // For bidirectional edges, add reverse direction
        if edge_type.is_bidirectional() {
            self.adjacency.get_mut(&target).unwrap().push((source, edge_idx));
        }
    }

    /// Get neighbors of a node
    pub fn get_neighbors(&self, node_id: &str) -> Vec<(&Node, &Edge)> {
        self.adjacency
            .get(node_id)
            .map(|neighbors| {
                neighbors
                    .iter()
                    .filter_map(|(neighbor_id, edge_idx)| {
                        let neighbor = self.nodes.get(neighbor_id)?;
                        let edge = self.edges.get(*edge_idx)?;
                        Some((neighbor, edge))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get neighbors by edge type
    pub fn get_neighbors_by_edge_type(&self, node_id: &str, edge_type: EdgeType) -> Vec<(&Node, &Edge)> {
        self.get_neighbors(node_id)
            .into_iter()
            .filter(|(_, edge)| edge.edge_type == edge_type)
            .collect()
    }

    /// Get neighbors by node type
    pub fn get_neighbors_by_node_type(&self, node_id: &str, node_type: NodeType) -> Vec<(&Node, &Edge)> {
        self.get_neighbors(node_id)
            .into_iter()
            .filter(|(node, _)| node.node_type == node_type)
            .collect()
    }

    /// Get all edges of a specific type
    pub fn get_edges_by_type(&self, edge_type: EdgeType) -> Vec<&Edge> {
        self.edges_by_type
            .get(&edge_type)
            .map(|indices| {
                indices
                    .iter()
                    .filter_map(|idx| self.edges.get(*idx))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Update timestamp and potentially decay edge weights
    pub fn tick(&mut self, timestamp: u64) {
        self.current_time = timestamp;
    }

    /// Get feature matrix for nodes of a specific type
    pub fn feature_matrix_by_type(&self, node_type: NodeType) -> (Array2<f64>, Vec<NodeId>) {
        let nodes: Vec<_> = self.get_nodes_by_type(node_type);
        if nodes.is_empty() {
            return (Array2::zeros((0, node_type.feature_dim())), Vec::new());
        }

        let dim = nodes[0].feature_vector().len();
        let mut matrix = Array2::zeros((nodes.len(), dim));
        let mut node_ids = Vec::with_capacity(nodes.len());

        for (i, node) in nodes.iter().enumerate() {
            let features = node.feature_vector();
            for (j, &val) in features.iter().enumerate() {
                if j < dim {
                    matrix[[i, j]] = val;
                }
            }
            node_ids.push(node.id.clone());
        }

        (matrix, node_ids)
    }

    /// Get adjacency matrix for edges of a specific type
    pub fn adjacency_matrix_by_type(&self, edge_type: EdgeType) -> (Array2<f64>, Vec<NodeId>) {
        let edges = self.get_edges_by_type(edge_type);
        if edges.is_empty() {
            return (Array2::zeros((0, 0)), Vec::new());
        }

        // Collect unique nodes involved in these edges
        let mut node_set = HashSet::new();
        for edge in &edges {
            node_set.insert(edge.source.clone());
            node_set.insert(edge.target.clone());
        }

        let node_ids: Vec<_> = node_set.into_iter().collect();
        let node_to_idx: HashMap<_, _> = node_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i))
            .collect();

        let n = node_ids.len();
        let mut matrix = Array2::zeros((n, n));

        for edge in edges {
            if let (Some(&i), Some(&j)) = (node_to_idx.get(&edge.source), node_to_idx.get(&edge.target)) {
                matrix[[i, j]] = edge.features.weight;
                if edge.edge_type.is_bidirectional() {
                    matrix[[j, i]] = edge.features.weight;
                }
            }
        }

        (matrix, node_ids)
    }

    /// Sample metapath instances starting from a node
    pub fn sample_metapath(&self, start_node: &str, metapath: &Metapath, num_samples: usize) -> Vec<MetapathInstance> {
        let mut instances = Vec::new();

        if !self.nodes.contains_key(start_node) {
            return instances;
        }

        // Check if start node matches metapath start type
        if let Some(start_type) = metapath.start_type() {
            if self.nodes.get(start_node).map(|n| n.node_type) != Some(start_type) {
                return instances;
            }
        }

        // Random walk sampling
        for _ in 0..num_samples {
            let mut path = vec![start_node.to_string()];
            let mut current = start_node.to_string();
            let mut weight = 1.0;
            let mut valid = true;

            for (i, (expected_type, edge_type)) in metapath.path.iter().skip(1).enumerate() {
                if let Some(required_edge) = metapath.path[i].1 {
                    let neighbors = self.get_neighbors_by_edge_type(&current, required_edge);

                    // Filter by expected node type
                    let valid_neighbors: Vec<_> = neighbors
                        .into_iter()
                        .filter(|(node, _)| node.node_type == *expected_type)
                        .collect();

                    if valid_neighbors.is_empty() {
                        valid = false;
                        break;
                    }

                    // Sample a neighbor (weighted by edge weight)
                    let total_weight: f64 = valid_neighbors.iter().map(|(_, e)| e.features.weight).sum();
                    let mut rng_val = rand::random::<f64>() * total_weight;

                    let mut chosen = None;
                    for (node, edge) in &valid_neighbors {
                        rng_val -= edge.features.weight;
                        if rng_val <= 0.0 {
                            chosen = Some((node, edge));
                            break;
                        }
                    }

                    if let Some((node, edge)) = chosen {
                        path.push(node.id.clone());
                        current = node.id.clone();
                        weight *= edge.features.weight;
                    } else {
                        valid = false;
                        break;
                    }
                }
            }

            if valid && path.len() == metapath.path.len() {
                instances.push(MetapathInstance::with_weight(metapath.clone(), path, weight));
            }
        }

        instances
    }

    /// Get graph statistics
    pub fn stats(&self) -> GraphStats {
        let node_count = self.nodes.len();
        let edge_count = self.edges.len();

        let mut nodes_per_type = HashMap::new();
        for (node_type, ids) in &self.nodes_by_type {
            nodes_per_type.insert(*node_type, ids.len());
        }

        let mut edges_per_type = HashMap::new();
        for (edge_type, indices) in &self.edges_by_type {
            edges_per_type.insert(*edge_type, indices.len());
        }

        let max_possible_edges = node_count * (node_count - 1) / 2;
        let density = if max_possible_edges > 0 {
            edge_count as f64 / max_possible_edges as f64
        } else {
            0.0
        };

        let avg_degree = if node_count > 0 {
            (2 * edge_count) as f64 / node_count as f64
        } else {
            0.0
        };

        GraphStats {
            node_count,
            edge_count,
            nodes_per_type,
            edges_per_type,
            density,
            avg_degree,
        }
    }

    /// Take a snapshot of the current graph state
    pub fn snapshot(&self) -> GraphSnapshot {
        GraphSnapshot {
            timestamp: self.current_time,
            node_count: self.nodes.len(),
            edge_count: self.edges.len(),
            nodes_by_type: self.nodes_by_type.iter()
                .map(|(t, ids)| (*t, ids.len()))
                .collect(),
            edges_by_type: self.edges_by_type.iter()
                .map(|(t, indices)| (*t, indices.len()))
                .collect(),
        }
    }

    /// Get the schema
    pub fn schema(&self) -> &GraphSchema {
        &self.schema
    }

    /// Get number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get number of edges
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }
}

/// Graph statistics
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub nodes_per_type: HashMap<NodeType, usize>,
    pub edges_per_type: HashMap<EdgeType, usize>,
    pub density: f64,
    pub avg_degree: f64,
}

/// Snapshot of graph state at a point in time
#[derive(Debug, Clone)]
pub struct GraphSnapshot {
    pub timestamp: u64,
    pub node_count: usize,
    pub edge_count: usize,
    pub nodes_by_type: HashMap<NodeType, usize>,
    pub edges_by_type: HashMap<EdgeType, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> HeterogeneousGraph {
        let schema = GraphSchema::trading_schema();
        let mut graph = HeterogeneousGraph::new(schema);

        // Add asset nodes
        let btc_features = super::super::AssetFeatures::new(50000.0, 1_000_000.0, 0.02);
        let eth_features = super::super::AssetFeatures::new(3000.0, 500_000.0, 0.03);
        let sol_features = super::super::AssetFeatures::new(100.0, 200_000.0, 0.05);

        graph.add_node("BTCUSDT", NodeType::Asset, btc_features.into());
        graph.add_node("ETHUSDT", NodeType::Asset, eth_features.into());
        graph.add_node("SOLUSDT", NodeType::Asset, sol_features.into());

        // Add exchange node
        let bybit_features = super::super::ExchangeFeatures::new(5_000_000.0, 100, 0.95);
        graph.add_node("Bybit", NodeType::Exchange, bybit_features.into());

        // Add edges
        graph.add_edge("BTCUSDT", "ETHUSDT", EdgeType::Correlation, EdgeFeatures::with_correlation(0.85, 1000));
        graph.add_edge("BTCUSDT", "SOLUSDT", EdgeType::Correlation, EdgeFeatures::with_correlation(0.72, 1000));
        graph.add_edge("ETHUSDT", "SOLUSDT", EdgeType::Correlation, EdgeFeatures::with_correlation(0.78, 1000));
        graph.add_edge("BTCUSDT", "Bybit", EdgeType::TradesOn, EdgeFeatures::with_volume(1_000_000.0, 1000));
        graph.add_edge("ETHUSDT", "Bybit", EdgeType::TradesOn, EdgeFeatures::with_volume(500_000.0, 1000));

        graph
    }

    #[test]
    fn test_graph_creation() {
        let graph = create_test_graph();
        assert_eq!(graph.num_nodes(), 4);
        assert_eq!(graph.num_edges(), 5);
    }

    #[test]
    fn test_get_neighbors() {
        let graph = create_test_graph();
        let neighbors = graph.get_neighbors("BTCUSDT");
        assert!(!neighbors.is_empty());
    }

    #[test]
    fn test_get_neighbors_by_type() {
        let graph = create_test_graph();
        let corr_neighbors = graph.get_neighbors_by_edge_type("BTCUSDT", EdgeType::Correlation);
        assert_eq!(corr_neighbors.len(), 2);

        let exchange_neighbors = graph.get_neighbors_by_edge_type("BTCUSDT", EdgeType::TradesOn);
        assert_eq!(exchange_neighbors.len(), 1);
    }

    #[test]
    fn test_feature_matrix() {
        let graph = create_test_graph();
        let (matrix, ids) = graph.feature_matrix_by_type(NodeType::Asset);
        assert_eq!(ids.len(), 3);
        assert_eq!(matrix.nrows(), 3);
    }

    #[test]
    fn test_graph_stats() {
        let graph = create_test_graph();
        let stats = graph.stats();
        assert_eq!(stats.node_count, 4);
        assert_eq!(stats.edge_count, 5);
        assert!(stats.avg_degree > 0.0);
    }
}
