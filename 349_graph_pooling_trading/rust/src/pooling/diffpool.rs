//! DiffPool (Differentiable Pooling) implementation
//!
//! Based on: "Hierarchical Graph Representation Learning with DiffPool"
//! https://arxiv.org/abs/1806.08804

use super::PoolingResult;
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// DiffPool layer
///
/// Learns soft cluster assignments through:
/// S = softmax(GNN_pool(A, X))
/// X' = S^T * Z  (pooled features)
/// A' = S^T * A * S  (pooled adjacency)
pub struct DiffPool {
    /// Number of output clusters
    n_clusters: usize,

    /// Weight matrix for cluster assignment [n_features, n_clusters]
    weight_pool: Array2<f64>,

    /// Weight matrix for embedding [n_features, n_features]
    weight_embed: Array2<f64>,

    /// Link prediction loss weight
    link_pred_weight: f64,

    /// Entropy loss weight
    entropy_weight: f64,
}

impl DiffPool {
    /// Create a new DiffPool layer
    pub fn new(n_clusters: usize) -> Self {
        Self::with_dim(n_clusters, 8) // Default feature dim
    }

    /// Create with specific feature dimension
    pub fn with_dim(n_clusters: usize, n_features: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        // Initialize weights with Xavier initialization
        let scale = (2.0 / (n_features + n_clusters) as f64).sqrt();

        let weight_pool = Array2::from_shape_fn((n_features, n_clusters), |_| {
            normal.sample(&mut rng) * scale
        });

        let weight_embed = Array2::from_shape_fn((n_features, n_features), |_| {
            normal.sample(&mut rng) * scale
        });

        Self {
            n_clusters,
            weight_pool,
            weight_embed,
            link_pred_weight: 1.0,
            entropy_weight: 1.0,
        }
    }

    /// Set loss weights
    pub fn with_loss_weights(mut self, link_pred: f64, entropy: f64) -> Self {
        self.link_pred_weight = link_pred;
        self.entropy_weight = entropy;
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

        // Resize weights if needed (for variable input sizes)
        let weight_pool = if n_features != self.weight_pool.nrows() {
            self.create_weight_matrix(n_features, self.n_clusters)
        } else {
            self.weight_pool.clone()
        };

        let weight_embed = if n_features != self.weight_embed.nrows() {
            self.create_weight_matrix(n_features, n_features)
        } else {
            self.weight_embed.clone()
        };

        // Graph convolution for embeddings: Z = A * X * W_embed
        let z = self.graph_conv(features, adjacency, &weight_embed);

        // Graph convolution for assignment: S_raw = A * X * W_pool
        let s_raw = self.graph_conv(features, adjacency, &weight_pool);

        // Softmax to get cluster assignments
        let s = self.softmax(&s_raw);

        // Pool features: X' = S^T * Z
        let pooled_features = s.t().dot(&z);

        // Pool adjacency: A' = S^T * A * S
        let pooled_adjacency = s.t().dot(adjacency).dot(&s);

        // Compute losses
        let link_pred_loss = self.link_prediction_loss(&s, adjacency);
        let entropy_loss = self.entropy_loss(&s);
        let total_loss =
            self.link_pred_weight * link_pred_loss + self.entropy_weight * entropy_loss;

        Ok(PoolingResult {
            pooled_features,
            pooled_adjacency,
            cluster_assignments: s,
            loss: total_loss,
        })
    }

    /// Simple graph convolution: A * X * W
    fn graph_conv(
        &self,
        features: &Array2<f64>,
        adjacency: &Array2<f64>,
        weight: &Array2<f64>,
    ) -> Array2<f64> {
        // Add self-loops
        let n = adjacency.nrows();
        let mut adj_with_self = adjacency.clone();
        for i in 0..n {
            adj_with_self[[i, i]] = 1.0;
        }

        // Normalize adjacency (D^-0.5 * A * D^-0.5)
        let degree: Array1<f64> = adj_with_self.sum_axis(Axis(1));
        let degree_inv_sqrt: Array1<f64> = degree.mapv(|d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 });

        let mut normalized_adj = adj_with_self.clone();
        for i in 0..n {
            for j in 0..n {
                normalized_adj[[i, j]] *= degree_inv_sqrt[i] * degree_inv_sqrt[j];
            }
        }

        // Propagate: A_norm * X
        let propagated = normalized_adj.dot(features);

        // Transform: (A_norm * X) * W
        propagated.dot(weight)
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

    /// Link prediction loss: ||A - S * S^T||_F^2
    ///
    /// Encourages nodes that are connected to be in the same cluster
    fn link_prediction_loss(&self, s: &Array2<f64>, adjacency: &Array2<f64>) -> f64 {
        let predicted = s.dot(&s.t());
        let diff = adjacency - &predicted;
        diff.iter().map(|&x| x * x).sum::<f64>() / (s.nrows() * s.nrows()) as f64
    }

    /// Entropy loss: sum(-S * log(S))
    ///
    /// Encourages soft assignments to become hard (low entropy)
    fn entropy_loss(&self, s: &Array2<f64>) -> f64 {
        let eps = 1e-10;
        -s.iter()
            .map(|&p| {
                let p_safe = p.max(eps);
                p_safe * p_safe.ln()
            })
            .sum::<f64>()
            / s.nrows() as f64
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
    fn test_diffpool_forward() {
        let n_nodes = 10;
        let n_features = 8;
        let n_clusters = 3;

        let features = Array2::from_shape_fn((n_nodes, n_features), |(i, j)| {
            ((i + j) as f64) / 100.0
        });

        let mut adjacency = Array2::<f64>::zeros((n_nodes, n_nodes));
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                if i != j && (i as i32 - j as i32).abs() <= 2 {
                    adjacency[[i, j]] = 1.0;
                }
            }
        }

        let pool = DiffPool::with_dim(n_clusters, n_features);
        let result = pool.forward(&features, &adjacency).unwrap();

        assert_eq!(result.pooled_features.nrows(), n_clusters);
        assert_eq!(result.pooled_features.ncols(), n_features);
        assert_eq!(result.pooled_adjacency.nrows(), n_clusters);
        assert_eq!(result.cluster_assignments.nrows(), n_nodes);
        assert_eq!(result.cluster_assignments.ncols(), n_clusters);

        // Check that assignments sum to 1 for each node
        for i in 0..n_nodes {
            let sum: f64 = result.cluster_assignments.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax() {
        let pool = DiffPool::new(3);
        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0]).unwrap();
        let s = pool.softmax(&x);

        // Each row should sum to 1
        for i in 0..2 {
            let sum: f64 = s.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}
