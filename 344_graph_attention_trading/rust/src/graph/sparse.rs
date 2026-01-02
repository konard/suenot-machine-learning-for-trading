//! Sparse graph representation using CSR format
//!
//! Efficient storage and operations for sparse asset graphs.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Compressed Sparse Row (CSR) graph representation
///
/// Efficient for graphs where not all pairs of nodes are connected.
/// Stores only existing edges, reducing memory usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseGraph {
    /// Number of nodes
    n_nodes: usize,
    /// Row pointers: indptr[i] to indptr[i+1] gives edge range for node i
    indptr: Vec<usize>,
    /// Column indices: target node for each edge
    indices: Vec<usize>,
    /// Edge weights (optional)
    data: Vec<f64>,
    /// Node labels (e.g., asset symbols)
    labels: Vec<String>,
}

impl SparseGraph {
    /// Create a new sparse graph
    pub fn new(n_nodes: usize) -> Self {
        Self {
            n_nodes,
            indptr: vec![0; n_nodes + 1],
            indices: Vec::new(),
            data: Vec::new(),
            labels: (0..n_nodes).map(|i| format!("node_{}", i)).collect(),
        }
    }

    /// Create from dense adjacency matrix
    pub fn from_dense(adjacency: &Array2<f64>) -> Self {
        let n = adjacency.nrows();
        let mut indptr = vec![0usize; n + 1];
        let mut indices = Vec::new();
        let mut data = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if adjacency[[i, j]] != 0.0 {
                    indices.push(j);
                    data.push(adjacency[[i, j]]);
                }
            }
            indptr[i + 1] = indices.len();
        }

        Self {
            n_nodes: n,
            indptr,
            indices,
            data,
            labels: (0..n).map(|i| format!("node_{}", i)).collect(),
        }
    }

    /// Create from edge list
    pub fn from_edges(n_nodes: usize, edges: &[(usize, usize, f64)]) -> Self {
        // Sort edges by source node
        let mut sorted_edges = edges.to_vec();
        sorted_edges.sort_by_key(|e| e.0);

        let mut indptr = vec![0usize; n_nodes + 1];
        let mut indices = Vec::with_capacity(edges.len());
        let mut data = Vec::with_capacity(edges.len());

        for (src, dst, weight) in sorted_edges {
            indices.push(dst);
            data.push(weight);
            indptr[src + 1] = indices.len();
        }

        // Fill in empty rows
        for i in 1..=n_nodes {
            if indptr[i] == 0 {
                indptr[i] = indptr[i - 1];
            }
        }

        Self {
            n_nodes,
            indptr,
            indices,
            data,
            labels: (0..n_nodes).map(|i| format!("node_{}", i)).collect(),
        }
    }

    /// Set node labels
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        assert_eq!(labels.len(), self.n_nodes);
        self.labels = labels;
        self
    }

    /// Get number of nodes
    pub fn num_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Get number of edges
    pub fn num_edges(&self) -> usize {
        self.indices.len()
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node: usize) -> &[usize] {
        let start = self.indptr[node];
        let end = self.indptr[node + 1];
        &self.indices[start..end]
    }

    /// Get edge weights for a node's outgoing edges
    pub fn edge_weights(&self, node: usize) -> &[f64] {
        let start = self.indptr[node];
        let end = self.indptr[node + 1];
        &self.data[start..end]
    }

    /// Get degree (number of outgoing edges) for a node
    pub fn degree(&self, node: usize) -> usize {
        self.indptr[node + 1] - self.indptr[node]
    }

    /// Check if edge exists
    pub fn has_edge(&self, src: usize, dst: usize) -> bool {
        self.neighbors(src).contains(&dst)
    }

    /// Get edge weight (returns 0 if edge doesn't exist)
    pub fn get_edge_weight(&self, src: usize, dst: usize) -> f64 {
        let neighbors = self.neighbors(src);
        let weights = self.edge_weights(src);

        for (i, &neighbor) in neighbors.iter().enumerate() {
            if neighbor == dst {
                return weights[i];
            }
        }
        0.0
    }

    /// Convert to dense adjacency matrix
    pub fn to_dense(&self) -> Array2<f64> {
        let mut adj = Array2::zeros((self.n_nodes, self.n_nodes));

        for i in 0..self.n_nodes {
            for (j_idx, &j) in self.neighbors(i).iter().enumerate() {
                adj[[i, j]] = self.edge_weights(i)[j_idx];
            }
        }

        adj
    }

    /// Get adjacency list representation
    pub fn adjacency_list(&self) -> Vec<Vec<(usize, f64)>> {
        (0..self.n_nodes)
            .map(|i| {
                self.neighbors(i)
                    .iter()
                    .zip(self.edge_weights(i).iter())
                    .map(|(&n, &w)| (n, w))
                    .collect()
            })
            .collect()
    }

    /// Compute node degrees
    pub fn degrees(&self) -> Array1<f64> {
        Array1::from_iter((0..self.n_nodes).map(|i| self.degree(i) as f64))
    }

    /// Get graph density
    pub fn density(&self) -> f64 {
        let max_edges = self.n_nodes * (self.n_nodes - 1);
        if max_edges == 0 {
            0.0
        } else {
            self.num_edges() as f64 / max_edges as f64
        }
    }

    /// Check if graph is symmetric
    pub fn is_symmetric(&self) -> bool {
        for i in 0..self.n_nodes {
            for &j in self.neighbors(i) {
                if !self.has_edge(j, i) {
                    return false;
                }
            }
        }
        true
    }

    /// Make graph symmetric by adding reverse edges
    pub fn make_symmetric(&self) -> Self {
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();

        for i in 0..self.n_nodes {
            for (j_idx, &j) in self.neighbors(i).iter().enumerate() {
                let w = self.edge_weights(i)[j_idx];
                edges.push((i, j, w));
                if !self.has_edge(j, i) {
                    edges.push((j, i, w));
                }
            }
        }

        Self::from_edges(self.n_nodes, &edges).with_labels(self.labels.clone())
    }

    /// Add self-loops
    pub fn add_self_loops(&self, weight: f64) -> Self {
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();

        // Add existing edges
        for i in 0..self.n_nodes {
            for (j_idx, &j) in self.neighbors(i).iter().enumerate() {
                edges.push((i, j, self.edge_weights(i)[j_idx]));
            }
        }

        // Add self-loops
        for i in 0..self.n_nodes {
            if !self.has_edge(i, i) {
                edges.push((i, i, weight));
            }
        }

        Self::from_edges(self.n_nodes, &edges).with_labels(self.labels.clone())
    }

    /// Normalize edge weights (row-wise)
    pub fn normalize(&self) -> Self {
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();

        for i in 0..self.n_nodes {
            let neighbors = self.neighbors(i);
            let weights = self.edge_weights(i);
            let sum: f64 = weights.iter().sum();

            if sum > 0.0 {
                for (j_idx, &j) in neighbors.iter().enumerate() {
                    edges.push((i, j, weights[j_idx] / sum));
                }
            }
        }

        Self::from_edges(self.n_nodes, &edges).with_labels(self.labels.clone())
    }

    /// Get subgraph induced by node subset
    pub fn subgraph(&self, nodes: &[usize]) -> Self {
        let n = nodes.len();
        let node_map: std::collections::HashMap<usize, usize> =
            nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();

        let mut edges: Vec<(usize, usize, f64)> = Vec::new();

        for (new_i, &old_i) in nodes.iter().enumerate() {
            for (j_idx, &old_j) in self.neighbors(old_i).iter().enumerate() {
                if let Some(&new_j) = node_map.get(&old_j) {
                    edges.push((new_i, new_j, self.edge_weights(old_i)[j_idx]));
                }
            }
        }

        let labels: Vec<String> = nodes.iter().map(|&i| self.labels[i].clone()).collect();

        Self::from_edges(n, &edges).with_labels(labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_graph_creation() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
        let graph = SparseGraph::from_edges(3, &edges);

        assert_eq!(graph.num_nodes(), 3);
        assert_eq!(graph.num_edges(), 3);
        assert!(graph.has_edge(0, 1));
        assert!(!graph.has_edge(1, 0));
    }

    #[test]
    fn test_from_dense() {
        let mut adj = Array2::zeros((3, 3));
        adj[[0, 1]] = 1.0;
        adj[[1, 2]] = 1.0;
        adj[[2, 0]] = 1.0;

        let graph = SparseGraph::from_dense(&adj);

        assert_eq!(graph.num_nodes(), 3);
        assert_eq!(graph.num_edges(), 3);
    }

    #[test]
    fn test_neighbors() {
        let edges = vec![(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)];
        let graph = SparseGraph::from_edges(3, &edges);

        let neighbors = graph.neighbors(0);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2));
    }

    #[test]
    fn test_symmetric() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0)];
        let graph = SparseGraph::from_edges(3, &edges);

        assert!(!graph.is_symmetric());

        let sym_graph = graph.make_symmetric();
        assert!(sym_graph.is_symmetric());
    }
}
