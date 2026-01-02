//! E-GNN Layer Implementation
//!
//! Core E(n) Equivariant layer that updates both node features and coordinates.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::Normal;

/// E(n) Equivariant Graph Neural Network Layer
#[derive(Debug, Clone)]
pub struct EGNNLayer {
    /// Hidden dimension
    hidden_dim: usize,

    /// Edge feature dimension
    edge_dim: usize,

    /// Whether to update coordinates
    update_coords: bool,

    /// Edge MLP weights
    edge_mlp_w1: Array2<f64>,
    edge_mlp_b1: Array1<f64>,
    edge_mlp_w2: Array2<f64>,
    edge_mlp_b2: Array1<f64>,

    /// Node MLP weights
    node_mlp_w1: Array2<f64>,
    node_mlp_b1: Array1<f64>,
    node_mlp_w2: Array2<f64>,
    node_mlp_b2: Array1<f64>,

    /// Coordinate MLP weights (if update_coords)
    coord_mlp_w1: Option<Array2<f64>>,
    coord_mlp_b1: Option<Array1<f64>>,
    coord_mlp_w2: Option<Array2<f64>>,
}

impl EGNNLayer {
    /// Create a new E-GNN layer
    pub fn new(hidden_dim: usize, edge_dim: usize, update_coords: bool) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        // Edge MLP: (2*hidden + 1 + edge_dim) -> hidden -> hidden
        let edge_input_dim = 2 * hidden_dim + 1 + edge_dim;
        let edge_mlp_w1 = Array2::from_shape_fn((edge_input_dim, hidden_dim), |_| rng.sample(normal));
        let edge_mlp_b1 = Array1::zeros(hidden_dim);
        let edge_mlp_w2 = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| rng.sample(normal));
        let edge_mlp_b2 = Array1::zeros(hidden_dim);

        // Node MLP: (2*hidden) -> hidden -> hidden
        let node_mlp_w1 = Array2::from_shape_fn((2 * hidden_dim, hidden_dim), |_| rng.sample(normal));
        let node_mlp_b1 = Array1::zeros(hidden_dim);
        let node_mlp_w2 = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| rng.sample(normal));
        let node_mlp_b2 = Array1::zeros(hidden_dim);

        // Coordinate MLP: hidden -> hidden -> 1
        let (coord_mlp_w1, coord_mlp_b1, coord_mlp_w2) = if update_coords {
            (
                Some(Array2::from_shape_fn((hidden_dim, hidden_dim), |_| rng.sample(normal))),
                Some(Array1::zeros(hidden_dim)),
                Some(Array2::from_shape_fn((hidden_dim, 1), |_| rng.sample(normal))),
            )
        } else {
            (None, None, None)
        };

        Self {
            hidden_dim,
            edge_dim,
            update_coords,
            edge_mlp_w1,
            edge_mlp_b1,
            edge_mlp_w2,
            edge_mlp_b2,
            node_mlp_w1,
            node_mlp_b1,
            node_mlp_w2,
            node_mlp_b2,
            coord_mlp_w1,
            coord_mlp_b1,
            coord_mlp_w2,
        }
    }

    /// Forward pass through the layer
    pub fn forward(
        &self,
        h: &Array2<f64>,      // Node features [N, hidden]
        x: &Array2<f64>,      // Coordinates [N, coord_dim]
        edge_index: &Array2<usize>, // [2, E]
        edge_attr: Option<&Array2<f64>>, // [E, edge_dim]
    ) -> (Array2<f64>, Array2<f64>) {
        let num_nodes = h.nrows();
        let num_edges = edge_index.ncols();
        let coord_dim = x.ncols();

        // Compute messages for each edge
        let mut messages = Array2::zeros((num_edges, self.hidden_dim));
        let mut coord_diffs = Array2::zeros((num_edges, coord_dim));

        for e in 0..num_edges {
            let i = edge_index[[0, e]];
            let j = edge_index[[1, e]];

            // Coordinate difference
            let diff: Array1<f64> = x.row(i).to_owned() - x.row(j).to_owned();
            coord_diffs.row_mut(e).assign(&diff);

            // Squared distance (invariant scalar)
            let radial: f64 = diff.iter().map(|d| d * d).sum();

            // Build edge input: [h_i, h_j, radial, edge_attr]
            let mut edge_input = Vec::with_capacity(2 * self.hidden_dim + 1 + self.edge_dim);
            edge_input.extend(h.row(i).iter());
            edge_input.extend(h.row(j).iter());
            edge_input.push(radial);

            if let Some(attr) = edge_attr {
                edge_input.extend(attr.row(e).iter());
            } else {
                edge_input.extend(vec![0.0; self.edge_dim]);
            }

            // Edge MLP forward
            let edge_input = Array1::from_vec(edge_input);
            let msg = self.edge_mlp_forward(&edge_input);
            messages.row_mut(e).assign(&msg);
        }

        // Update coordinates (equivariant)
        let x_new = if self.update_coords {
            let mut coord_updates = Array2::zeros((num_nodes, coord_dim));
            let mut coord_counts = vec![0usize; num_nodes];

            for e in 0..num_edges {
                let i = edge_index[[0, e]];
                let msg = messages.row(e);

                // Coordinate weight from message
                let weight = self.coord_mlp_forward(&msg.to_owned());

                // Weighted coordinate difference
                for d in 0..coord_dim {
                    coord_updates[[i, d]] += coord_diffs[[e, d]] * weight;
                }
                coord_counts[i] += 1;
            }

            // Average and add to original coordinates
            let mut x_new = x.clone();
            for i in 0..num_nodes {
                if coord_counts[i] > 0 {
                    for d in 0..coord_dim {
                        x_new[[i, d]] += coord_updates[[i, d]] / coord_counts[i] as f64;
                    }
                }
            }
            x_new
        } else {
            x.clone()
        };

        // Aggregate messages for each node
        let mut msg_agg = Array2::zeros((num_nodes, self.hidden_dim));
        let mut msg_counts = vec![0usize; num_nodes];

        for e in 0..num_edges {
            let i = edge_index[[0, e]];
            for d in 0..self.hidden_dim {
                msg_agg[[i, d]] += messages[[e, d]];
            }
            msg_counts[i] += 1;
        }

        // Update node features
        let mut h_new = Array2::zeros((num_nodes, self.hidden_dim));
        for i in 0..num_nodes {
            // Concatenate [h_i, aggregated_messages]
            let mut node_input = Vec::with_capacity(2 * self.hidden_dim);
            node_input.extend(h.row(i).iter());

            if msg_counts[i] > 0 {
                for d in 0..self.hidden_dim {
                    node_input.push(msg_agg[[i, d]] / msg_counts[i] as f64);
                }
            } else {
                node_input.extend(vec![0.0; self.hidden_dim]);
            }

            let node_input = Array1::from_vec(node_input);
            let h_update = self.node_mlp_forward(&node_input);

            // Residual connection
            for d in 0..self.hidden_dim {
                h_new[[i, d]] = h[[i, d]] + h_update[d];
            }
        }

        (h_new, x_new)
    }

    /// Edge MLP forward pass
    fn edge_mlp_forward(&self, input: &Array1<f64>) -> Array1<f64> {
        // First layer
        let z1 = input.dot(&self.edge_mlp_w1) + &self.edge_mlp_b1;
        let a1 = z1.mapv(|x| silu(x));

        // Second layer
        let z2 = a1.dot(&self.edge_mlp_w2) + &self.edge_mlp_b2;
        z2.mapv(|x| silu(x))
    }

    /// Node MLP forward pass
    fn node_mlp_forward(&self, input: &Array1<f64>) -> Array1<f64> {
        // First layer
        let z1 = input.dot(&self.node_mlp_w1) + &self.node_mlp_b1;
        let a1 = z1.mapv(|x| silu(x));

        // Second layer
        let z2 = a1.dot(&self.node_mlp_w2) + &self.node_mlp_b2;
        z2
    }

    /// Coordinate MLP forward pass
    fn coord_mlp_forward(&self, input: &Array1<f64>) -> f64 {
        if let (Some(w1), Some(b1), Some(w2)) = (&self.coord_mlp_w1, &self.coord_mlp_b1, &self.coord_mlp_w2) {
            let z1 = input.dot(w1) + b1;
            let a1 = z1.mapv(|x| silu(x));
            let z2 = a1.dot(w2);
            z2[0]
        } else {
            0.0
        }
    }
}

/// SiLU activation function
fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_creation() {
        let layer = EGNNLayer::new(64, 3, true);
        assert_eq!(layer.hidden_dim, 64);
        assert!(layer.update_coords);
    }

    #[test]
    fn test_layer_forward() {
        let layer = EGNNLayer::new(8, 2, true);

        let h = Array2::from_shape_fn((3, 8), |_| 0.1);
        let x = Array2::from_shape_fn((3, 3), |(i, j)| (i + j) as f64);
        let edge_index = Array2::from_shape_vec((2, 4), vec![0, 0, 1, 2, 1, 2, 0, 1]).unwrap();
        let edge_attr = Array2::from_shape_fn((4, 2), |_| 0.5);

        let (h_new, x_new) = layer.forward(&h, &x, &edge_index, Some(&edge_attr));

        assert_eq!(h_new.shape(), &[3, 8]);
        assert_eq!(x_new.shape(), &[3, 3]);
    }

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 1e-10);
        assert!(silu(1.0) > 0.0);
        assert!(silu(-1.0) < 0.0);
    }
}
