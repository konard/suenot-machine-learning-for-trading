//! SAGPool (Self-Attention Graph Pooling) implementation
//!
//! Based on: "Self-Attention Graph Pooling"
//! https://arxiv.org/abs/1904.08082

use super::PoolingResult;
use ndarray::{Array1, Array2, Axis};
use rand::Rng;

/// SAGPool layer
///
/// Selects top-k nodes based on self-attention scores.
/// Unlike DiffPool, this is a sparse pooling method.
pub struct SAGPool {
    /// Pooling ratio (fraction of nodes to keep)
    ratio: f64,

    /// Attention weight vector
    attention_weights: Array1<f64>,
}

impl SAGPool {
    /// Create a new SAGPool layer
    ///
    /// # Arguments
    /// * `ratio` - Fraction of nodes to keep (0 < ratio <= 1)
    pub fn new(ratio: f64) -> Self {
        assert!(ratio > 0.0 && ratio <= 1.0, "Ratio must be in (0, 1]");

        Self {
            ratio,
            attention_weights: Array1::zeros(0), // Will be initialized on first forward
        }
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
        let k = ((n_nodes as f64 * self.ratio).ceil() as usize).max(1);

        // Initialize attention weights if needed
        let attention = if self.attention_weights.len() != n_features {
            Self::init_attention(n_features)
        } else {
            self.attention_weights.clone()
        };

        // Graph convolution to get node embeddings
        let z = self.graph_conv(features, adjacency);

        // Compute attention scores: score = tanh(Z * a)
        let scores: Array1<f64> = z.dot(&attention).mapv(|x| x.tanh());

        // Select top-k nodes
        let (selected_indices, importance_scores) = self.top_k_indices(&scores, k);

        // Build pooled features: X' = Z[idx] * sigmoid(scores[idx])
        let mut pooled_features = Array2::<f64>::zeros((k, n_features));
        for (new_idx, &orig_idx) in selected_indices.iter().enumerate() {
            let gate = Self::sigmoid(importance_scores[new_idx]);
            for j in 0..n_features {
                pooled_features[[new_idx, j]] = z[[orig_idx, j]] * gate;
            }
        }

        // Build pooled adjacency (induced subgraph)
        let mut pooled_adjacency = Array2::<f64>::zeros((k, k));
        for (i, &orig_i) in selected_indices.iter().enumerate() {
            for (j, &orig_j) in selected_indices.iter().enumerate() {
                pooled_adjacency[[i, j]] = adjacency[[orig_i, orig_j]];
            }
        }

        // Build soft cluster assignments (one-hot for selected nodes)
        // This is a simplification - SAGPool doesn't have soft assignments
        let mut cluster_assignments = Array2::<f64>::zeros((n_nodes, k));
        for (cluster_idx, &node_idx) in selected_indices.iter().enumerate() {
            cluster_assignments[[node_idx, cluster_idx]] = 1.0;
        }

        // Loss is negative mean of selected scores (higher scores = better)
        let loss = -importance_scores.iter().sum::<f64>() / k as f64;

        Ok(PoolingResult {
            pooled_features,
            pooled_adjacency,
            cluster_assignments,
            loss,
        })
    }

    /// Get the importance scores for all nodes
    pub fn compute_importance(
        &self,
        features: &Array2<f64>,
        adjacency: &Array2<f64>,
    ) -> Array1<f64> {
        let n_features = features.ncols();

        let attention = if self.attention_weights.len() != n_features {
            Self::init_attention(n_features)
        } else {
            self.attention_weights.clone()
        };

        let z = self.graph_conv(features, adjacency);
        z.dot(&attention).mapv(|x| x.tanh())
    }

    /// Simple graph convolution
    fn graph_conv(&self, features: &Array2<f64>, adjacency: &Array2<f64>) -> Array2<f64> {
        let n = adjacency.nrows();

        // Add self-loops
        let mut adj = adjacency.clone();
        for i in 0..n {
            adj[[i, i]] = 1.0;
        }

        // Normalize
        let degree: Array1<f64> = adj.sum_axis(Axis(1));
        let degree_inv_sqrt: Array1<f64> = degree.mapv(|d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 });

        for i in 0..n {
            for j in 0..n {
                adj[[i, j]] *= degree_inv_sqrt[i] * degree_inv_sqrt[j];
            }
        }

        adj.dot(features)
    }

    /// Get top-k indices and their scores
    fn top_k_indices(&self, scores: &Array1<f64>, k: usize) -> (Vec<usize>, Vec<f64>) {
        let mut indexed: Vec<(usize, f64)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let k = k.min(indexed.len());
        let selected: Vec<(usize, f64)> = indexed.into_iter().take(k).collect();

        // Sort by original index for consistent ordering
        let mut sorted = selected.clone();
        sorted.sort_by_key(|(idx, _)| *idx);

        let indices: Vec<usize> = sorted.iter().map(|(idx, _)| *idx).collect();
        let importance: Vec<f64> = sorted.iter().map(|(_, score)| *score).collect();

        (indices, importance)
    }

    /// Initialize attention weights
    fn init_attention(n_features: usize) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let scale = (1.0 / n_features as f64).sqrt();
        Array1::from_shape_fn(n_features, |_| rng.gen::<f64>() * scale * 2.0 - scale)
    }

    /// Sigmoid activation
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sagpool_forward() {
        let n_nodes = 10;
        let n_features = 8;

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

        let pool = SAGPool::new(0.5);
        let result = pool.forward(&features, &adjacency).unwrap();

        // Should keep 5 nodes (50% of 10)
        assert_eq!(result.pooled_features.nrows(), 5);
        assert_eq!(result.pooled_adjacency.nrows(), 5);
    }

    #[test]
    fn test_importance_scores() {
        let features = Array2::from_shape_fn((5, 4), |(i, _)| i as f64);
        let adjacency = Array2::from_shape_fn((5, 5), |(i, j)| {
            if i != j { 1.0 } else { 0.0 }
        });

        let pool = SAGPool::new(0.5);
        let scores = pool.compute_importance(&features, &adjacency);

        assert_eq!(scores.len(), 5);
        // Scores should be in [-1, 1] due to tanh
        for &s in scores.iter() {
            assert!(s >= -1.0 && s <= 1.0);
        }
    }
}
