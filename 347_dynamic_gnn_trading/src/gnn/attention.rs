//! Attention mechanisms for GNN

use ndarray::{Array1, Array2, Axis};
use rand::Rng;

use super::leaky_relu;

/// Graph Attention mechanism (GAT style)
#[derive(Debug, Clone)]
pub struct GraphAttention {
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Weight matrices for each head
    weights: Vec<Array2<f64>>,
    /// Attention vector for each head
    attention: Vec<Array1<f64>>,
    /// Leaky ReLU negative slope
    alpha: f64,
}

impl GraphAttention {
    /// Create a new graph attention layer
    pub fn new(hidden_dim: usize, num_heads: usize) -> Self {
        let mut rng = rand::thread_rng();
        let head_dim = hidden_dim / num_heads;
        let scale = (2.0 / (hidden_dim + head_dim) as f64).sqrt();

        let weights: Vec<Array2<f64>> = (0..num_heads)
            .map(|_| {
                Array2::from_shape_fn((hidden_dim, head_dim), |_| rng.gen_range(-scale..scale))
            })
            .collect();

        let attention: Vec<Array1<f64>> = (0..num_heads)
            .map(|_| Array1::from_shape_fn(2 * head_dim, |_| rng.gen_range(-scale..scale)))
            .collect();

        Self {
            hidden_dim,
            num_heads,
            weights,
            attention,
            alpha: 0.2,
        }
    }

    /// Forward pass with attention
    pub fn forward(&self, features: &Array2<f64>, adjacency: &Array2<f64>) -> Array2<f64> {
        let n = features.nrows();
        let head_dim = self.hidden_dim / self.num_heads;
        let mut outputs = Vec::new();

        // Process each attention head
        for head in 0..self.num_heads {
            // Linear transformation: H' = H * W
            let h = features.dot(&self.weights[head]);

            // Compute attention coefficients
            let attention_matrix = self.compute_attention(&h, adjacency, head);

            // Apply attention: H'' = attention * H'
            let attended = attention_matrix.dot(&h);
            outputs.push(attended);
        }

        // Concatenate heads
        let mut result = Array2::zeros((n, self.hidden_dim));
        for (head_idx, head_output) in outputs.iter().enumerate() {
            for i in 0..n {
                for j in 0..head_dim {
                    result[[i, head_idx * head_dim + j]] = head_output[[i, j]];
                }
            }
        }

        result
    }

    /// Compute attention coefficients
    fn compute_attention(
        &self,
        features: &Array2<f64>,
        adjacency: &Array2<f64>,
        head: usize,
    ) -> Array2<f64> {
        let n = features.nrows();
        let head_dim = features.ncols();

        // Compute attention scores for all pairs
        let mut scores = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                // Only compute for adjacent nodes
                if adjacency[[i, j]] > 0.0 || i == j {
                    // Concatenate features: [h_i || h_j]
                    let mut concat = Array1::zeros(2 * head_dim);
                    for k in 0..head_dim {
                        concat[k] = features[[i, k]];
                        concat[head_dim + k] = features[[j, k]];
                    }

                    // Compute attention score: a^T * [h_i || h_j]
                    let score: f64 = concat
                        .iter()
                        .zip(self.attention[head].iter())
                        .map(|(c, a)| c * a)
                        .sum();

                    scores[[i, j]] = leaky_relu(score, self.alpha);
                } else {
                    scores[[i, j]] = f64::NEG_INFINITY;
                }
            }
        }

        // Softmax over neighbors
        let mut attention = Array2::zeros((n, n));
        for i in 0..n {
            // Get max for numerical stability
            let max_score = scores
                .row(i)
                .iter()
                .filter(|&&x| x != f64::NEG_INFINITY)
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);

            // Compute exp and sum
            let mut exp_sum = 0.0;
            for j in 0..n {
                if scores[[i, j]] != f64::NEG_INFINITY {
                    let exp_val = (scores[[i, j]] - max_score).exp();
                    attention[[i, j]] = exp_val;
                    exp_sum += exp_val;
                }
            }

            // Normalize
            if exp_sum > 0.0 {
                for j in 0..n {
                    attention[[i, j]] /= exp_sum;
                }
            }
        }

        attention
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        self.num_heads * (self.weights[0].len() + self.attention[0].len())
    }
}

/// Multi-head attention for sequence-like graph data
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    /// Model dimension
    pub d_model: usize,
    /// Number of heads
    pub num_heads: usize,
    /// Dimension per head
    pub d_k: usize,
    /// Query projection
    w_q: Array2<f64>,
    /// Key projection
    w_k: Array2<f64>,
    /// Value projection
    w_v: Array2<f64>,
    /// Output projection
    w_o: Array2<f64>,
}

impl MultiHeadAttention {
    /// Create new multi-head attention
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        let d_k = d_model / num_heads;
        let mut rng = rand::thread_rng();
        let scale = (2.0 / d_model as f64).sqrt();

        let w_q = Array2::from_shape_fn((d_model, d_model), |_| rng.gen_range(-scale..scale));
        let w_k = Array2::from_shape_fn((d_model, d_model), |_| rng.gen_range(-scale..scale));
        let w_v = Array2::from_shape_fn((d_model, d_model), |_| rng.gen_range(-scale..scale));
        let w_o = Array2::from_shape_fn((d_model, d_model), |_| rng.gen_range(-scale..scale));

        Self {
            d_model,
            num_heads,
            d_k,
            w_q,
            w_k,
            w_v,
            w_o,
        }
    }

    /// Forward pass
    pub fn forward(&self, query: &Array2<f64>, key: &Array2<f64>, value: &Array2<f64>) -> Array2<f64> {
        let n = query.nrows();

        // Linear projections
        let q = query.dot(&self.w_q);
        let k = key.dot(&self.w_k);
        let v = value.dot(&self.w_v);

        // Scaled dot-product attention
        // Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        let scale = (self.d_k as f64).sqrt();
        let scores = q.dot(&k.t()) / scale;

        // Softmax
        let mut attention = Array2::zeros((n, n));
        for i in 0..n {
            let max = scores.row(i).iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp: Array1<f64> = scores.row(i).mapv(|x| (x - max).exp());
            let sum: f64 = exp.sum();
            for j in 0..n {
                attention[[i, j]] = exp[j] / sum;
            }
        }

        // Apply attention to values
        let attended = attention.dot(&v);

        // Output projection
        attended.dot(&self.w_o)
    }

    /// Self-attention: query = key = value
    pub fn self_attention(&self, x: &Array2<f64>) -> Array2<f64> {
        self.forward(x, x, x)
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        self.w_q.len() + self.w_k.len() + self.w_v.len() + self.w_o.len()
    }
}

/// Edge attention for learning edge importance
#[derive(Debug, Clone)]
pub struct EdgeAttention {
    /// Edge feature dimension
    pub edge_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Edge transformation weights
    edge_weights: Array2<f64>,
    /// Attention vector
    attention: Array1<f64>,
}

impl EdgeAttention {
    /// Create new edge attention
    pub fn new(edge_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (edge_dim + hidden_dim) as f64).sqrt();

        let edge_weights = Array2::from_shape_fn((edge_dim, hidden_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let attention = Array1::from_shape_fn(hidden_dim, |_| rng.gen_range(-scale..scale));

        Self {
            edge_dim,
            hidden_dim,
            edge_weights,
            attention,
        }
    }

    /// Compute edge attention weights
    pub fn compute_weights(&self, edge_features: &Array2<f64>) -> Array1<f64> {
        let n = edge_features.nrows();

        // Transform edge features
        let transformed = edge_features.dot(&self.edge_weights);

        // Compute attention scores
        let mut scores = Array1::zeros(n);
        for i in 0..n {
            let score: f64 = transformed
                .row(i)
                .iter()
                .zip(self.attention.iter())
                .map(|(t, a)| t * a)
                .sum();
            scores[i] = score;
        }

        // Softmax
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp = scores.mapv(|x| (x - max).exp());
        let sum: f64 = exp.sum();
        exp / sum
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        self.edge_weights.len() + self.attention.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_attention() {
        let attention = GraphAttention::new(8, 2);
        let features = Array2::from_shape_fn((4, 8), |_| rand::random::<f64>());
        let adjacency = Array2::from_shape_fn((4, 4), |(i, j)| {
            if i != j { 0.5 } else { 0.0 }
        });

        let output = attention.forward(&features, &adjacency);
        assert_eq!(output.shape(), &[4, 8]);
    }

    #[test]
    fn test_multi_head_attention() {
        let mha = MultiHeadAttention::new(8, 2);
        let x = Array2::from_shape_fn((4, 8), |_| rand::random::<f64>());

        let output = mha.self_attention(&x);
        assert_eq!(output.shape(), &[4, 8]);
    }

    #[test]
    fn test_edge_attention() {
        let edge_attn = EdgeAttention::new(4, 8);
        let edge_features = Array2::from_shape_fn((5, 4), |_| rand::random::<f64>());

        let weights = edge_attn.compute_weights(&edge_features);
        assert_eq!(weights.len(), 5);
        assert!((weights.sum() - 1.0).abs() < 0.001);
    }
}
