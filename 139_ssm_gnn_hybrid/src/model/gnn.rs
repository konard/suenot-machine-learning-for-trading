//! Graph Attention Network layer implementation.
//!
//! Implements GAT-style attention for cross-asset message passing
//! on correlation or sector graphs.

use rand::Rng;
use rand_distr::StandardNormal;

/// Graph Attention Layer (GAT).
///
/// Computes attention-weighted message passing between connected nodes
/// in the asset relationship graph.
pub struct GraphAttentionLayer {
    pub d_in: usize,
    pub d_out: usize,
    /// Linear transform W (d_in x d_out)
    pub w: Vec<Vec<f64>>,
    /// Source attention vector (d_out,)
    pub a_src: Vec<f64>,
    /// Target attention vector (d_out,)
    pub a_dst: Vec<f64>,
}

impl GraphAttentionLayer {
    /// Create a new GAT layer with Xavier initialization.
    pub fn new(d_in: usize, d_out: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (d_in + d_out) as f64).sqrt();

        let w = (0..d_in)
            .map(|_| {
                (0..d_out)
                    .map(|_| rng.sample::<f64, _>(StandardNormal) * scale)
                    .collect()
            })
            .collect();

        let a_src = (0..d_out)
            .map(|_| rng.sample::<f64, _>(StandardNormal) * 0.01)
            .collect();

        let a_dst = (0..d_out)
            .map(|_| rng.sample::<f64, _>(StandardNormal) * 0.01)
            .collect();

        Self {
            d_in,
            d_out,
            w,
            a_src,
            a_dst,
        }
    }

    /// Forward pass with graph attention.
    ///
    /// # Arguments
    /// * `node_features` - (n_nodes, d_in) node feature matrix
    /// * `edge_index` - (2, n_edges) source-target edge indices
    /// * `edge_weight` - Optional (n_edges,) edge weights
    ///
    /// # Returns
    /// (n_nodes, d_out) updated node features
    pub fn forward(
        &self,
        node_features: &[Vec<f64>],
        edge_index: &[Vec<usize>],
        edge_weight: Option<&[f64]>,
    ) -> Vec<Vec<f64>> {
        let n_nodes = node_features.len();
        let src_idx = &edge_index[0];
        let dst_idx = &edge_index[1];
        let n_edges = src_idx.len();

        // Linear transform: h = features @ W
        let h: Vec<Vec<f64>> = node_features
            .iter()
            .map(|feat| {
                (0..self.d_out)
                    .map(|o| {
                        (0..self.d_in)
                            .map(|i| feat[i] * self.w[i][o])
                            .sum::<f64>()
                    })
                    .collect()
            })
            .collect();

        // Attention coefficients: e_k = LeakyReLU(a_src . h[i] + a_dst . h[j])
        let mut e = vec![0.0_f64; n_edges];
        for k in 0..n_edges {
            let i = src_idx[k];
            let j = dst_idx[k];
            let score: f64 = (0..self.d_out)
                .map(|d| self.a_src[d] * h[i][d] + self.a_dst[d] * h[j][d])
                .sum();
            // LeakyReLU with negative_slope = 0.2
            e[k] = if score > 0.0 { score } else { 0.2 * score };
        }

        // Softmax per destination node
        let mut alpha = vec![0.0_f64; n_edges];
        for node in 0..n_nodes {
            let mut max_e = f64::NEG_INFINITY;
            let mut indices = Vec::new();

            for k in 0..n_edges {
                if dst_idx[k] == node {
                    indices.push(k);
                    if e[k] > max_e {
                        max_e = e[k];
                    }
                }
            }

            if indices.is_empty() {
                continue;
            }

            let mut sum = 0.0;
            for &k in &indices {
                let mut val = (e[k] - max_e).exp();
                if let Some(ew) = edge_weight {
                    val *= ew[k];
                }
                alpha[k] = val;
                sum += val;
            }

            for &k in &indices {
                alpha[k] /= sum + 1e-10;
            }
        }

        // Aggregate: out[j] = sum_k alpha[k] * h[src[k]] where dst[k] == j
        let mut out = vec![vec![0.0_f64; self.d_out]; n_nodes];
        for k in 0..n_edges {
            let j = dst_idx[k];
            let i = src_idx[k];
            for d in 0..self.d_out {
                out[j][d] += alpha[k] * h[i][d];
            }
        }

        // ELU activation
        for node in out.iter_mut() {
            for val in node.iter_mut() {
                if *val < 0.0 {
                    *val = val.exp() - 1.0;
                }
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gat_forward() {
        let gat = GraphAttentionLayer::new(4, 8);

        let features = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        // Fully connected graph
        let edge_index = vec![
            vec![0, 0, 1, 1, 2, 2],
            vec![1, 2, 0, 2, 0, 1],
        ];

        let output = gat.forward(&features, &edge_index, None);
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].len(), 8);
    }
}
