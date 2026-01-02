//! Graph Attention Layer implementation
//!
//! Single layer of Graph Attention Network.

use super::attention::{elu, MultiHeadAttention};
use crate::graph::SparseGraph;
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Graph Attention Layer
///
/// Applies multi-head attention to aggregate neighbor information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAttentionLayer {
    /// Input dimension
    in_features: usize,
    /// Output dimension (per head)
    out_features: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Weight matrices for each head
    weights: Vec<Array2<f64>>,
    /// Multi-head attention
    attention: MultiHeadAttention,
    /// Bias terms
    bias: Option<Array1<f64>>,
    /// Whether to concatenate heads (False for last layer)
    concat: bool,
    /// Dropout rate
    dropout: f64,
    /// ELU alpha parameter
    alpha: f64,
}

impl GraphAttentionLayer {
    /// Create a new Graph Attention Layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        num_heads: usize,
        concat: bool,
        dropout: f64,
    ) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization for weights
        let scale = (2.0 / (in_features + out_features) as f64).sqrt();

        let weights: Vec<Array2<f64>> = (0..num_heads)
            .map(|_| {
                Array2::from_shape_fn((in_features, out_features), |_| {
                    rng.gen_range(-scale..scale)
                })
            })
            .collect();

        let attention = if concat {
            MultiHeadAttention::new(out_features, num_heads, 0.2, dropout)
        } else {
            MultiHeadAttention::new(out_features, num_heads, 0.2, dropout).with_averaging()
        };

        let output_dim = if concat {
            out_features * num_heads
        } else {
            out_features
        };

        let bias = Some(Array1::zeros(output_dim));

        Self {
            in_features,
            out_features,
            num_heads,
            weights,
            attention,
            bias,
            concat,
            dropout,
            alpha: 1.0,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &Array2<f64>, graph: &SparseGraph) -> Array2<f64> {
        let n = x.nrows();

        // Apply linear transformation for each head
        let z_heads: Vec<Array2<f64>> = self
            .weights
            .iter()
            .map(|w| x.dot(w))
            .collect();

        // Compute attention for each head
        let mut head_outputs: Vec<Array2<f64>> = Vec::with_capacity(self.num_heads);

        for (head_idx, z) in z_heads.iter().enumerate() {
            let mut output = Array2::zeros((n, self.out_features));

            for i in 0..n {
                let neighbors = graph.neighbors(i);

                if neighbors.is_empty() {
                    // No neighbors: use self-loop
                    output.row_mut(i).assign(&z.row(i));
                    continue;
                }

                // Compute attention scores
                let attention = self.attention.heads[head_idx]
                    .compute_attention(i, z, graph);

                // Aggregate neighbor features
                let mut aggregated = Array1::zeros(self.out_features);
                for (idx, &j) in neighbors.iter().enumerate() {
                    aggregated.scaled_add(attention[idx], &z.row(j));
                }

                output.row_mut(i).assign(&aggregated);
            }

            head_outputs.push(output);
        }

        // Combine heads
        let output = if self.concat {
            // Concatenate all heads
            let combined_dim = self.out_features * self.num_heads;
            let mut combined = Array2::zeros((n, combined_dim));

            for (head_idx, head_out) in head_outputs.iter().enumerate() {
                let start = head_idx * self.out_features;
                let end = start + self.out_features;
                combined
                    .slice_mut(ndarray::s![.., start..end])
                    .assign(head_out);
            }
            combined
        } else {
            // Average all heads
            let mut avg = Array2::zeros((n, self.out_features));
            for head_out in &head_outputs {
                avg = avg + head_out;
            }
            avg / self.num_heads as f64
        };

        // Add bias and apply activation
        let output = if let Some(ref bias) = self.bias {
            output + bias
        } else {
            output
        };

        // Apply ELU activation
        output.mapv(|x| elu(x, self.alpha))
    }

    /// Get attention weights for visualization
    pub fn get_attention_weights(&self, x: &Array2<f64>, graph: &SparseGraph) -> Array2<f64> {
        // Transform features
        let z = x.dot(&self.weights[0]);

        // Compute and average attention from all heads
        let attention_matrices = self.attention.compute_attention(&z, graph);
        self.attention.aggregate_attention(&attention_matrices)
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        let weight_params = self.in_features * self.out_features * self.num_heads;
        let attention_params = 2 * self.out_features * self.num_heads;
        let bias_params = if self.concat {
            self.out_features * self.num_heads
        } else {
            self.out_features
        };

        weight_params + attention_params + bias_params
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        if self.concat {
            self.out_features * self.num_heads
        } else {
            self.out_features
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_layer_forward() {
        let n = 5;
        let in_dim = 10;
        let out_dim = 8;
        let heads = 2;

        let layer = GraphAttentionLayer::new(in_dim, out_dim, heads, true, 0.0);
        let x = Array2::random((n, in_dim), Uniform::new(-1.0, 1.0));

        // Create simple graph
        let adj = crate::graph::GraphBuilder::sample_adjacency(n);
        let graph = SparseGraph::from_dense(&adj);

        let output = layer.forward(&x, &graph);

        // Check output shape
        assert_eq!(output.nrows(), n);
        assert_eq!(output.ncols(), out_dim * heads); // Concatenated
    }

    #[test]
    fn test_layer_averaging() {
        let n = 5;
        let in_dim = 10;
        let out_dim = 8;
        let heads = 2;

        let layer = GraphAttentionLayer::new(in_dim, out_dim, heads, false, 0.0);
        let x = Array2::random((n, in_dim), Uniform::new(-1.0, 1.0));

        let adj = crate::graph::GraphBuilder::sample_adjacency(n);
        let graph = SparseGraph::from_dense(&adj);

        let output = layer.forward(&x, &graph);

        // Check output shape (averaged)
        assert_eq!(output.nrows(), n);
        assert_eq!(output.ncols(), out_dim);
    }

    #[test]
    fn test_attention_weights() {
        let n = 5;
        let in_dim = 10;
        let out_dim = 8;
        let heads = 2;

        let layer = GraphAttentionLayer::new(in_dim, out_dim, heads, true, 0.0);
        let x = Array2::random((n, in_dim), Uniform::new(-1.0, 1.0));

        let adj = crate::graph::GraphBuilder::sample_adjacency(n);
        let graph = SparseGraph::from_dense(&adj);

        let attention = layer.get_attention_weights(&x, &graph);

        // Check attention shape
        assert_eq!(attention.nrows(), n);
        assert_eq!(attention.ncols(), n);

        // Row sums should be close to 1 (for nodes with neighbors)
        for i in 0..n {
            let row_sum: f64 = attention.row(i).sum();
            if graph.degree(i) > 0 {
                assert!((row_sum - 1.0).abs() < 0.1);
            }
        }
    }
}
