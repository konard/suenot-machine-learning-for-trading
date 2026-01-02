//! Attention mechanisms for GAT
//!
//! Implements single and multi-head attention for graph data.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::graph::SparseGraph;

/// Single attention head
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionHead {
    /// Attention vector for computing scores
    attention_vector: Array1<f64>,
    /// Negative slope for LeakyReLU
    negative_slope: f64,
    /// Dropout rate
    dropout_rate: f64,
}

impl AttentionHead {
    /// Create a new attention head
    pub fn new(dim: usize, negative_slope: f64, dropout_rate: f64) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization for attention vector
        let scale = (2.0 / (2.0 * dim as f64)).sqrt();
        let attention_vector =
            Array1::from_iter((0..2 * dim).map(|_| rng.gen_range(-scale..scale)));

        Self {
            attention_vector,
            negative_slope,
            dropout_rate,
        }
    }

    /// Compute attention score between two nodes
    pub fn compute_score(&self, zi: &Array1<f64>, zj: &Array1<f64>) -> f64 {
        // Concatenate zi and zj
        let concat = ndarray::concatenate![Axis(0), zi.view(), zj.view()];

        // Compute score
        let score = self.attention_vector.dot(&concat);

        // Apply LeakyReLU
        leaky_relu(score, self.negative_slope)
    }

    /// Compute attention coefficients for a node
    pub fn compute_attention(
        &self,
        node_id: usize,
        z: &Array2<f64>,
        graph: &SparseGraph,
    ) -> Array1<f64> {
        let neighbors = graph.neighbors(node_id);

        if neighbors.is_empty() {
            return Array1::zeros(0);
        }

        // Compute attention scores
        let zi = z.row(node_id);
        let scores: Vec<f64> = neighbors
            .iter()
            .map(|&j| self.compute_score(&zi.to_owned(), &z.row(j).to_owned()))
            .collect();

        // Apply softmax
        softmax(&scores)
    }

    /// Compute full attention matrix
    pub fn compute_attention_matrix(
        &self,
        z: &Array2<f64>,
        graph: &SparseGraph,
    ) -> Array2<f64> {
        let n = z.nrows();
        let mut attention = Array2::zeros((n, n));

        for i in 0..n {
            let neighbors = graph.neighbors(i);
            if neighbors.is_empty() {
                continue;
            }

            let zi = z.row(i);
            let scores: Vec<f64> = neighbors
                .iter()
                .map(|&j| self.compute_score(&zi.to_owned(), &z.row(j).to_owned()))
                .collect();

            let alpha = softmax(&scores);

            for (idx, &j) in neighbors.iter().enumerate() {
                attention[[i, j]] = alpha[idx];
            }
        }

        attention
    }

    /// Apply dropout to attention weights
    pub fn apply_dropout(&self, attention: &mut Array1<f64>, training: bool) {
        if !training || self.dropout_rate <= 0.0 {
            return;
        }

        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - self.dropout_rate);

        for a in attention.iter_mut() {
            if rng.gen::<f64>() < self.dropout_rate {
                *a = 0.0;
            } else {
                *a *= scale;
            }
        }
    }
}

/// Multi-head attention mechanism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiHeadAttention {
    /// Attention heads
    heads: Vec<AttentionHead>,
    /// Whether to concatenate or average heads
    concatenate: bool,
}

impl MultiHeadAttention {
    /// Create multi-head attention
    pub fn new(dim: usize, num_heads: usize, negative_slope: f64, dropout_rate: f64) -> Self {
        let heads = (0..num_heads)
            .map(|_| AttentionHead::new(dim, negative_slope, dropout_rate))
            .collect();

        Self {
            heads,
            concatenate: true,
        }
    }

    /// Create with averaging (for final layer)
    pub fn with_averaging(mut self) -> Self {
        self.concatenate = false;
        self
    }

    /// Get number of heads
    pub fn num_heads(&self) -> usize {
        self.heads.len()
    }

    /// Compute attention for all heads
    pub fn compute_attention(
        &self,
        z: &Array2<f64>,
        graph: &SparseGraph,
    ) -> Vec<Array2<f64>> {
        self.heads
            .iter()
            .map(|head| head.compute_attention_matrix(z, graph))
            .collect()
    }

    /// Aggregate attention from all heads
    pub fn aggregate_attention(&self, attention_matrices: &[Array2<f64>]) -> Array2<f64> {
        if attention_matrices.is_empty() {
            return Array2::zeros((0, 0));
        }

        let n = attention_matrices[0].nrows();
        let mut avg = Array2::zeros((n, n));

        for att in attention_matrices {
            avg = avg + att;
        }

        avg / attention_matrices.len() as f64
    }
}

/// LeakyReLU activation
fn leaky_relu(x: f64, negative_slope: f64) -> f64 {
    if x >= 0.0 {
        x
    } else {
        negative_slope * x
    }
}

/// Softmax function
fn softmax(scores: &[f64]) -> Array1<f64> {
    if scores.is_empty() {
        return Array1::zeros(0);
    }

    // Numerical stability: subtract max
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum: f64 = exp_scores.iter().sum();

    Array1::from_iter(exp_scores.iter().map(|&e| e / sum))
}

/// ELU activation
pub fn elu(x: f64, alpha: f64) -> f64 {
    if x >= 0.0 {
        x
    } else {
        alpha * (x.exp() - 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leaky_relu() {
        assert_eq!(leaky_relu(1.0, 0.2), 1.0);
        assert_eq!(leaky_relu(-1.0, 0.2), -0.2);
        assert_eq!(leaky_relu(0.0, 0.2), 0.0);
    }

    #[test]
    fn test_softmax() {
        let scores = vec![1.0, 2.0, 3.0];
        let result = softmax(&scores);

        // Sum should be 1
        assert!((result.sum() - 1.0).abs() < 1e-10);

        // Higher score should have higher probability
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_attention_head() {
        let head = AttentionHead::new(4, 0.2, 0.0);

        let zi = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let zj = Array1::from_vec(vec![4.0, 3.0, 2.0, 1.0]);

        let score = head.compute_score(&zi, &zj);
        // Score should be finite
        assert!(score.is_finite());
    }
}
