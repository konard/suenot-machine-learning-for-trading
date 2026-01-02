//! Positional Encoding for graphs
//!
//! Provides various positional encoding schemes for graph nodes.

use super::config::PositionalEncodingType;
use crate::graph::market_graph::MarketGraph;
use ndarray::{Array1, Array2};
use rand::Rng;

/// Positional encoding for graph nodes
#[derive(Debug, Clone)]
pub struct PositionalEncoding {
    /// Type of encoding
    pub encoding_type: PositionalEncodingType,
    /// Number of encoding dimensions
    pub num_dims: usize,
    /// Random walk steps (for RWPE)
    pub walk_steps: usize,
}

impl PositionalEncoding {
    /// Create new positional encoding
    pub fn new(encoding_type: PositionalEncodingType, num_dims: usize) -> Self {
        Self {
            encoding_type,
            num_dims,
            walk_steps: 16,
        }
    }

    /// Set random walk steps
    pub fn with_walk_steps(mut self, steps: usize) -> Self {
        self.walk_steps = steps;
        self
    }

    /// Compute positional encoding for a graph
    pub fn encode(&self, graph: &MarketGraph) -> Array2<f64> {
        match self.encoding_type {
            PositionalEncodingType::None => {
                Array2::zeros((graph.num_nodes(), self.num_dims))
            }
            PositionalEncodingType::Laplacian => {
                self.laplacian_pe(graph)
            }
            PositionalEncodingType::RandomWalk => {
                self.random_walk_pe(graph)
            }
            PositionalEncodingType::LaplacianRW => {
                let lap = self.laplacian_pe(graph);
                let rw = self.random_walk_pe(graph);
                // Concatenate and truncate to num_dims
                self.concat_and_truncate(&lap, &rw)
            }
            PositionalEncodingType::Degree => {
                self.degree_pe(graph)
            }
            PositionalEncodingType::Learnable => {
                // Return random initialized encoding (would be learned during training)
                self.random_pe(graph.num_nodes())
            }
        }
    }

    /// Laplacian positional encoding using eigenvectors
    fn laplacian_pe(&self, graph: &MarketGraph) -> Array2<f64> {
        let n = graph.num_nodes();
        if n == 0 {
            return Array2::zeros((0, self.num_dims));
        }

        let laplacian = graph.laplacian_matrix();

        // Simple power iteration to approximate eigenvectors
        // In production, use proper eigenvalue decomposition
        let mut pe = Array2::zeros((n, self.num_dims));
        let mut rng = rand::thread_rng();

        for k in 0..self.num_dims.min(n) {
            // Initialize random vector
            let mut v: Array1<f64> = Array1::from_shape_fn(n, |_| rng.gen::<f64>() - 0.5);

            // Power iteration (simplified)
            for _ in 0..50 {
                // Multiply by (I - L) to get smallest eigenvectors
                let identity = Array2::eye(n);
                let m = &identity - &laplacian * 0.5;
                v = m.dot(&v);

                // Normalize
                let norm = v.mapv(|x| x * x).sum().sqrt();
                if norm > 1e-10 {
                    v.mapv_inplace(|x| x / norm);
                }

                // Orthogonalize against previous vectors
                for j in 0..k {
                    let prev = pe.column(j);
                    let dot: f64 = v.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
                    for i in 0..n {
                        v[i] -= dot * pe[[i, j]];
                    }
                }
            }

            // Store eigenvector
            for i in 0..n {
                pe[[i, k]] = v[i];
            }
        }

        pe
    }

    /// Random walk positional encoding
    fn random_walk_pe(&self, graph: &MarketGraph) -> Array2<f64> {
        let n = graph.num_nodes();
        if n == 0 {
            return Array2::zeros((0, self.num_dims));
        }

        // Compute transition matrix
        let adj = graph.adjacency_matrix();
        let mut transition = Array2::zeros((n, n));

        for i in 0..n {
            let degree = graph.degree(i) as f64;
            if degree > 0.0 {
                for j in 0..n {
                    transition[[i, j]] = adj[[i, j]] / degree;
                }
            } else {
                // Self-loop for isolated nodes
                transition[[i, i]] = 1.0;
            }
        }

        // Compute landing probabilities for different walk lengths
        let mut pe = Array2::zeros((n, self.num_dims));
        let mut power = Array2::eye(n);

        for k in 0..self.num_dims.min(self.walk_steps) {
            power = power.dot(&transition);

            // Diagonal of power matrix = probability of returning to same node
            for i in 0..n {
                pe[[i, k]] = power[[i, i]];
            }
        }

        pe
    }

    /// Degree-based positional encoding
    fn degree_pe(&self, graph: &MarketGraph) -> Array2<f64> {
        let n = graph.num_nodes();
        let mut pe = Array2::zeros((n, self.num_dims));

        for i in 0..n {
            let degree = graph.degree(i) as f64;
            let max_degree = (0..n).map(|j| graph.degree(j)).max().unwrap_or(1) as f64;
            let normalized_degree = degree / max_degree.max(1.0);

            // Encode degree in different frequencies (like sinusoidal PE)
            for k in 0..self.num_dims {
                let freq = (k + 1) as f64;
                if k % 2 == 0 {
                    pe[[i, k]] = (normalized_degree * freq * std::f64::consts::PI).sin();
                } else {
                    pe[[i, k]] = (normalized_degree * freq * std::f64::consts::PI).cos();
                }
            }
        }

        pe
    }

    /// Random positional encoding (for learnable embeddings)
    fn random_pe(&self, num_nodes: usize) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let std = (1.0 / self.num_dims as f64).sqrt();
        Array2::from_shape_fn((num_nodes, self.num_dims), |_| {
            (rng.gen::<f64>() - 0.5) * 2.0 * std
        })
    }

    /// Concatenate two encodings and truncate to num_dims
    fn concat_and_truncate(&self, a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let n = a.nrows();
        let dims_a = self.num_dims / 2;
        let dims_b = self.num_dims - dims_a;

        let mut result = Array2::zeros((n, self.num_dims));

        for i in 0..n {
            for j in 0..dims_a.min(a.ncols()) {
                result[[i, j]] = a[[i, j]];
            }
            for j in 0..dims_b.min(b.ncols()) {
                result[[i, dims_a + j]] = b[[i, j]];
            }
        }

        result
    }

    /// Add positional encoding to node features
    pub fn add_to_features(&self, features: &Array2<f64>, graph: &MarketGraph) -> Array2<f64> {
        let pe = self.encode(graph);
        let n = features.nrows();
        let total_dim = features.ncols() + pe.ncols();

        let mut result = Array2::zeros((n, total_dim));

        for i in 0..n {
            for j in 0..features.ncols() {
                result[[i, j]] = features[[i, j]];
            }
            for j in 0..pe.ncols() {
                result[[i, features.ncols() + j]] = pe[[i, j]];
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builder::create_test_graph;

    #[test]
    fn test_laplacian_pe() {
        let graph = create_test_graph();
        let pe = PositionalEncoding::new(PositionalEncodingType::Laplacian, 4);
        let encoding = pe.encode(&graph);

        assert_eq!(encoding.nrows(), graph.num_nodes());
        assert_eq!(encoding.ncols(), 4);
    }

    #[test]
    fn test_random_walk_pe() {
        let graph = create_test_graph();
        let pe = PositionalEncoding::new(PositionalEncodingType::RandomWalk, 4);
        let encoding = pe.encode(&graph);

        assert_eq!(encoding.nrows(), graph.num_nodes());
        assert_eq!(encoding.ncols(), 4);
    }

    #[test]
    fn test_degree_pe() {
        let graph = create_test_graph();
        let pe = PositionalEncoding::new(PositionalEncodingType::Degree, 4);
        let encoding = pe.encode(&graph);

        assert_eq!(encoding.nrows(), graph.num_nodes());
        assert_eq!(encoding.ncols(), 4);
    }
}
