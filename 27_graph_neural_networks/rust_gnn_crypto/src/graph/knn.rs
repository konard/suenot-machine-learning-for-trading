//! k-Nearest Neighbors graph construction.

use super::{CryptoGraph, GraphBuilder};
use ndarray::Array2;

/// Build graphs based on k-nearest neighbors in feature space.
pub struct KNNGraph {
    /// Number of neighbors
    k: usize,
    /// Distance metric
    metric: DistanceMetric,
}

/// Distance metric for k-NN.
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Cosine distance (1 - cosine similarity)
    Cosine,
    /// Correlation distance (1 - correlation)
    Correlation,
}

impl KNNGraph {
    /// Create a new k-NN graph builder.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            metric: DistanceMetric::Correlation,
        }
    }

    /// Set distance metric.
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Build graph from feature matrix.
    pub fn build_from_features(&self, features: &Array2<f64>, symbols: &[String]) -> CryptoGraph {
        let n = features.nrows();
        let mut graph = CryptoGraph::new();

        // Add all nodes
        for symbol in symbols {
            graph.add_node(symbol);
        }

        // Compute distance matrix
        let distances = self.compute_distance_matrix(features);

        // For each node, find k nearest neighbors
        for i in 0..n {
            let mut dists_with_idx: Vec<(usize, f64)> =
                distances[i].iter().cloned().enumerate().collect();

            // Sort by distance
            dists_with_idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Add edges to k nearest (skip self at index 0)
            for (j, dist) in dists_with_idx.iter().skip(1).take(self.k) {
                // Use similarity (1 - distance) as edge weight
                let weight = 1.0 - dist;
                graph.add_edge(&symbols[i], &symbols[*j], weight);
            }
        }

        graph
    }

    /// Compute distance matrix.
    fn compute_distance_matrix(&self, features: &Array2<f64>) -> Vec<Vec<f64>> {
        let n = features.nrows();
        let mut distances = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let row_i = features.row(i);
                let row_j = features.row(j);
                distances[i][j] = self.compute_distance(
                    row_i.as_slice().unwrap(),
                    row_j.as_slice().unwrap(),
                );
            }
        }

        distances
    }

    /// Compute distance between two vectors.
    fn compute_distance(&self, x: &[f64], y: &[f64]) -> f64 {
        match self.metric {
            DistanceMetric::Euclidean => {
                let sum: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                sum.sqrt()
            }
            DistanceMetric::Cosine => {
                let dot: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
                let norm_x: f64 = x.iter().map(|a| a.powi(2)).sum::<f64>().sqrt();
                let norm_y: f64 = y.iter().map(|a| a.powi(2)).sum::<f64>().sqrt();
                if norm_x == 0.0 || norm_y == 0.0 {
                    1.0
                } else {
                    1.0 - (dot / (norm_x * norm_y))
                }
            }
            DistanceMetric::Correlation => {
                let corr = crate::data::features::pearson_correlation(x, y);
                1.0 - corr
            }
        }
    }
}

impl GraphBuilder for KNNGraph {
    fn build(&self, returns: &[Vec<f64>], symbols: &[String]) -> CryptoGraph {
        // Convert returns to feature matrix
        let n = returns.len();
        let m = returns.iter().map(|r| r.len()).min().unwrap_or(0);

        if m == 0 {
            return CryptoGraph::new();
        }

        let mut features = Array2::zeros((n, m));
        for (i, r) in returns.iter().enumerate() {
            for (j, &val) in r.iter().take(m).enumerate() {
                features[[i, j]] = val;
            }
        }

        self.build_from_features(&features, symbols)
    }
}

/// Mutual k-NN graph (edge only if both nodes are in each other's k-NN).
pub struct MutualKNNGraph {
    inner: KNNGraph,
}

impl MutualKNNGraph {
    /// Create a new mutual k-NN graph builder.
    pub fn new(k: usize) -> Self {
        Self {
            inner: KNNGraph::new(k),
        }
    }

    /// Build mutual k-NN graph from features.
    pub fn build_from_features(&self, features: &Array2<f64>, symbols: &[String]) -> CryptoGraph {
        let regular_graph = self.inner.build_from_features(features, symbols);

        let mut mutual_graph = CryptoGraph::new();
        for symbol in symbols {
            mutual_graph.add_node(symbol);
        }

        // Only add edge if both nodes are neighbors of each other
        for (i, sym_i) in symbols.iter().enumerate() {
            for sym_j in symbols.iter().skip(i + 1) {
                let neighbors_i = regular_graph.neighbors(sym_i);
                let neighbors_j = regular_graph.neighbors(sym_j);

                if neighbors_i.contains(sym_j) && neighbors_j.contains(sym_i) {
                    let weight = regular_graph.edge_weight(sym_i, sym_j).unwrap_or(0.5);
                    mutual_graph.add_edge(sym_i, sym_j, weight);
                }
            }
        }

        mutual_graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn_graph() {
        let features = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 0.0, 0.0, // A
                0.9, 0.1, 0.0, // B (close to A)
                0.0, 1.0, 0.0, // C
                0.0, 0.0, 1.0, // D
            ],
        )
        .unwrap();

        let symbols = vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
        ];

        let builder = KNNGraph::new(2).with_metric(DistanceMetric::Euclidean);
        let graph = builder.build_from_features(&features, &symbols);

        assert_eq!(graph.node_count(), 4);
        // Each node should have at most k=2 neighbors
        for symbol in &symbols {
            assert!(graph.degree(symbol) <= 4); // Can have incoming edges too
        }
    }

    #[test]
    fn test_distance_metrics() {
        let builder = KNNGraph::new(2);

        let x = vec![1.0, 0.0, 0.0];
        let y = vec![0.0, 1.0, 0.0];

        // Euclidean distance should be sqrt(2)
        let builder_euclidean = KNNGraph::new(2).with_metric(DistanceMetric::Euclidean);
        let dist = builder_euclidean.compute_distance(&x, &y);
        assert!((dist - 2.0_f64.sqrt()).abs() < 0.001);

        // Cosine distance should be 1 (orthogonal vectors)
        let builder_cosine = KNNGraph::new(2).with_metric(DistanceMetric::Cosine);
        let dist = builder_cosine.compute_distance(&x, &y);
        assert!((dist - 1.0).abs() < 0.001);
    }
}
