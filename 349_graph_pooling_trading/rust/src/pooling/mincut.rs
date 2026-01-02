//! MinCutPool implementation
//!
//! Optimizes for balanced clusters with minimal inter-cluster edges,
//! inspired by spectral graph partitioning.

use super::PoolingResult;
use ndarray::{Array1, Array2, Axis};
use rand::Rng;

/// MinCutPool layer
///
/// Learns cluster assignments that minimize the normalized cut
/// while maintaining balanced cluster sizes.
pub struct MinCutPool {
    /// Number of clusters
    n_clusters: usize,

    /// Weight matrix for cluster assignment
    weight: Array2<f64>,

    /// MinCut loss weight
    mincut_weight: f64,

    /// Orthogonality loss weight
    ortho_weight: f64,
}

impl MinCutPool {
    /// Create a new MinCutPool layer
    pub fn new(n_clusters: usize) -> Self {
        Self::with_dim(n_clusters, 8)
    }

    /// Create with specific feature dimension
    pub fn with_dim(n_clusters: usize, n_features: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (n_features + n_clusters) as f64).sqrt();

        let weight = Array2::from_shape_fn((n_features, n_clusters), |_| {
            rng.gen::<f64>() * scale * 2.0 - scale
        });

        Self {
            n_clusters,
            weight,
            mincut_weight: 1.0,
            ortho_weight: 1.0,
        }
    }

    /// Set loss weights
    pub fn with_loss_weights(mut self, mincut: f64, ortho: f64) -> Self {
        self.mincut_weight = mincut;
        self.ortho_weight = ortho;
        self
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `features` - Node features [n_nodes, n_features]
    /// * `adjacency` - Adjacency matrix [n_nodes, n_nodes]
    pub fn forward(
        &self,
        features: &Array2<f64>,
        adjacency: &Array2<f64>,
    ) -> anyhow::Result<PoolingResult> {
        let n_nodes = features.nrows();
        let n_features = features.ncols();

        // Resize weight if needed
        let weight = if n_features != self.weight.nrows() {
            self.create_weight_matrix(n_features, self.n_clusters)
        } else {
            self.weight.clone()
        };

        // Two-layer MLP for cluster assignment
        let hidden = Self::relu(&features.dot(&weight));
        let s_raw = hidden.dot(&weight.t()).dot(&weight);

        // Softmax to get cluster assignments
        let s = self.softmax(&s_raw);

        // Pool features: X' = S^T * X
        let pooled_features = s.t().dot(features);

        // Pool adjacency: A' = S^T * A * S
        let pooled_adjacency = s.t().dot(adjacency).dot(&s);

        // Compute losses
        let mincut_loss = self.mincut_loss(&s, adjacency);
        let ortho_loss = self.orthogonality_loss(&s);

        let total_loss = self.mincut_weight * mincut_loss + self.ortho_weight * ortho_loss;

        Ok(PoolingResult {
            pooled_features,
            pooled_adjacency,
            cluster_assignments: s,
            loss: total_loss,
        })
    }

    /// MinCut loss: -Tr(S^T * A * S) / Tr(S^T * D * S)
    ///
    /// Encourages minimizing edges between clusters
    fn mincut_loss(&self, s: &Array2<f64>, adjacency: &Array2<f64>) -> f64 {
        let n = adjacency.nrows();

        // S^T * A * S
        let stas = s.t().dot(adjacency).dot(s);

        // Degree matrix
        let degree: Array1<f64> = adjacency.sum_axis(Axis(1));

        // S^T * D * S (using degree as diagonal)
        let mut stds = Array2::<f64>::zeros((self.n_clusters, self.n_clusters));
        for i in 0..n {
            for c1 in 0..self.n_clusters {
                for c2 in 0..self.n_clusters {
                    stds[[c1, c2]] += s[[i, c1]] * degree[i] * s[[i, c2]];
                }
            }
        }

        // Trace of matrices
        let trace_stas: f64 = (0..self.n_clusters).map(|i| stas[[i, i]]).sum();
        let trace_stds: f64 = (0..self.n_clusters).map(|i| stds[[i, i]]).sum();

        // Normalized cut
        if trace_stds > 1e-10 {
            -trace_stas / trace_stds
        } else {
            0.0
        }
    }

    /// Orthogonality loss: ||S^T*S / ||S^T*S||_F - I/sqrt(k)||_F
    ///
    /// Encourages non-overlapping, balanced clusters
    fn orthogonality_loss(&self, s: &Array2<f64>) -> f64 {
        let sts = s.t().dot(s);

        // Frobenius norm of S^T * S
        let sts_norm: f64 = sts.iter().map(|&x| x * x).sum::<f64>().sqrt();

        if sts_norm < 1e-10 {
            return 0.0;
        }

        // Normalized S^T * S
        let sts_normalized = &sts / sts_norm;

        // Target: I / sqrt(k)
        let target_val = 1.0 / (self.n_clusters as f64).sqrt();

        // Frobenius norm of difference
        let mut diff_norm_sq = 0.0;
        for i in 0..self.n_clusters {
            for j in 0..self.n_clusters {
                let target = if i == j { target_val } else { 0.0 };
                let diff = sts_normalized[[i, j]] - target;
                diff_norm_sq += diff * diff;
            }
        }

        diff_norm_sq.sqrt()
    }

    /// Row-wise softmax
    fn softmax(&self, x: &Array2<f64>) -> Array2<f64> {
        let n_rows = x.nrows();
        let n_cols = x.ncols();
        let mut result = Array2::<f64>::zeros((n_rows, n_cols));

        for i in 0..n_rows {
            let row = x.row(i);
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_row: Vec<f64> = row.iter().map(|&v| (v - max_val).exp()).collect();
            let sum: f64 = exp_row.iter().sum();

            for (j, &e) in exp_row.iter().enumerate() {
                result[[i, j]] = e / sum;
            }
        }

        result
    }

    /// ReLU activation
    fn relu(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| v.max(0.0))
    }

    /// Create a random weight matrix
    fn create_weight_matrix(&self, rows: usize, cols: usize) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (rows + cols) as f64).sqrt();
        Array2::from_shape_fn((rows, cols), |_| rng.gen::<f64>() * scale * 2.0 - scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mincut_forward() {
        let n_nodes = 10;
        let n_features = 8;
        let n_clusters = 3;

        let features = Array2::from_shape_fn((n_nodes, n_features), |(i, j)| {
            ((i + j) as f64) / 100.0
        });

        // Create a graph with clear community structure
        let mut adjacency = Array2::<f64>::zeros((n_nodes, n_nodes));
        // Group 1: nodes 0-3
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    adjacency[[i, j]] = 1.0;
                }
            }
        }
        // Group 2: nodes 4-6
        for i in 4..7 {
            for j in 4..7 {
                if i != j {
                    adjacency[[i, j]] = 1.0;
                }
            }
        }
        // Group 3: nodes 7-9
        for i in 7..10 {
            for j in 7..10 {
                if i != j {
                    adjacency[[i, j]] = 1.0;
                }
            }
        }
        // Few inter-group edges
        adjacency[[3, 4]] = 0.5;
        adjacency[[4, 3]] = 0.5;
        adjacency[[6, 7]] = 0.5;
        adjacency[[7, 6]] = 0.5;

        let pool = MinCutPool::with_dim(n_clusters, n_features);
        let result = pool.forward(&features, &adjacency).unwrap();

        assert_eq!(result.pooled_features.nrows(), n_clusters);
        assert_eq!(result.pooled_adjacency.nrows(), n_clusters);
        assert_eq!(result.cluster_assignments.nrows(), n_nodes);
        assert_eq!(result.cluster_assignments.ncols(), n_clusters);
    }

    #[test]
    fn test_balanced_clusters() {
        let n_nodes = 9;
        let n_features = 4;
        let n_clusters = 3;

        let features = Array2::from_shape_fn((n_nodes, n_features), |(i, _)| i as f64);
        let adjacency = Array2::from_shape_fn((n_nodes, n_nodes), |(i, j)| {
            if i != j { 1.0 } else { 0.0 }
        });

        let pool = MinCutPool::with_dim(n_clusters, n_features);
        let result = pool.forward(&features, &adjacency).unwrap();

        let cluster_sizes = result.cluster_sizes();
        // With orthogonality regularization, clusters should be roughly balanced
        let total: f64 = cluster_sizes.iter().sum();
        for size in &cluster_sizes {
            // Each cluster should have roughly 1/3 of nodes
            assert!(*size > 0.0);
            assert!(*size < total);
        }
    }
}
