//! GNN layer implementations

use ndarray::{Array1, Array2, Axis};
use rand::Rng;

use super::relu;

/// Base GNN layer with message passing
#[derive(Debug, Clone)]
pub struct GNNLayer {
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Weight matrix for self-connection
    weights_self: Array2<f64>,
    /// Weight matrix for neighbor aggregation
    weights_neigh: Array2<f64>,
    /// Bias vector
    bias: Array1<f64>,
}

impl GNNLayer {
    /// Create a new GNN layer
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();

        let weights_self = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let weights_neigh = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let bias = Array1::zeros(output_dim);

        Self {
            input_dim,
            output_dim,
            weights_self,
            weights_neigh,
            bias,
        }
    }

    /// Forward pass
    pub fn forward(&self, features: &Array2<f64>, adjacency: &Array2<f64>) -> Array2<f64> {
        let n = features.nrows();

        // Normalize adjacency matrix (add self-loops and normalize)
        let adj_norm = self.normalize_adjacency(adjacency);

        // Self transformation
        let h_self = features.dot(&self.weights_self);

        // Neighbor aggregation: A * H * W_neigh
        let h_neigh = adj_norm.dot(features).dot(&self.weights_neigh);

        // Combine and apply activation
        let mut output = Array2::zeros((n, self.output_dim));
        for i in 0..n {
            for j in 0..self.output_dim {
                let val = h_self[[i, j]] + h_neigh[[i, j]] + self.bias[j];
                output[[i, j]] = relu(val);
            }
        }

        output
    }

    /// Normalize adjacency matrix with self-loops
    fn normalize_adjacency(&self, adjacency: &Array2<f64>) -> Array2<f64> {
        let n = adjacency.nrows();
        let mut adj = adjacency.clone();

        // Add self-loops
        for i in 0..n {
            adj[[i, i]] = 1.0;
        }

        // Compute degree matrix
        let degrees: Array1<f64> = adj.sum_axis(Axis(1));

        // D^(-1/2) * A * D^(-1/2) normalization
        let mut normalized = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if degrees[i] > 0.0 && degrees[j] > 0.0 {
                    normalized[[i, j]] =
                        adj[[i, j]] / (degrees[i].sqrt() * degrees[j].sqrt());
                }
            }
        }

        normalized
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        self.weights_self.len() + self.weights_neigh.len() + self.bias.len()
    }
}

/// Graph Convolutional Layer (GCN style)
#[derive(Debug, Clone)]
pub struct GraphConvLayer {
    /// Weight matrix
    weights: Array2<f64>,
    /// Bias
    bias: Array1<f64>,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
}

impl GraphConvLayer {
    /// Create a new graph convolutional layer
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();

        let weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let bias = Array1::zeros(output_dim);

        Self {
            weights,
            bias,
            input_dim,
            output_dim,
        }
    }

    /// Forward pass: H' = σ(Ã * H * W)
    pub fn forward(&self, features: &Array2<f64>, adjacency: &Array2<f64>) -> Array2<f64> {
        let adj_norm = self.symmetric_normalize(adjacency);
        let ah = adj_norm.dot(features);
        let ahw = ah.dot(&self.weights);

        // Add bias and apply ReLU
        let n = ahw.nrows();
        let mut output = Array2::zeros((n, self.output_dim));
        for i in 0..n {
            for j in 0..self.output_dim {
                output[[i, j]] = relu(ahw[[i, j]] + self.bias[j]);
            }
        }

        output
    }

    /// Symmetric normalization: D^(-1/2) * (A + I) * D^(-1/2)
    fn symmetric_normalize(&self, adjacency: &Array2<f64>) -> Array2<f64> {
        let n = adjacency.nrows();
        let mut adj = adjacency.clone();

        // Add self-loops
        for i in 0..n {
            adj[[i, i]] += 1.0;
        }

        // Compute degrees
        let degrees: Array1<f64> = adj.sum_axis(Axis(1));
        let d_inv_sqrt: Array1<f64> = degrees.mapv(|d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 });

        // Normalize
        let mut normalized = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                normalized[[i, j]] = d_inv_sqrt[i] * adj[[i, j]] * d_inv_sqrt[j];
            }
        }

        normalized
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        self.weights.len() + self.bias.len()
    }
}

/// Message Passing Layer with customizable aggregation
#[derive(Debug, Clone)]
pub struct MessagePassingLayer {
    /// Message transformation weights
    msg_weights: Array2<f64>,
    /// Update transformation weights
    update_weights: Array2<f64>,
    /// Bias
    bias: Array1<f64>,
    /// Input dimension
    pub input_dim: usize,
    /// Message dimension
    pub msg_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Aggregation type
    pub aggregation: AggregationType,
}

/// Type of aggregation for message passing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggregationType {
    Sum,
    Mean,
    Max,
}

impl MessagePassingLayer {
    /// Create a new message passing layer
    pub fn new(input_dim: usize, msg_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + msg_dim) as f64).sqrt();

        let msg_weights = Array2::from_shape_fn((input_dim, msg_dim), |_| {
            rng.gen_range(-scale..scale)
        });

        let scale2 = (2.0 / (input_dim + msg_dim + output_dim) as f64).sqrt();
        let update_weights = Array2::from_shape_fn((input_dim + msg_dim, output_dim), |_| {
            rng.gen_range(-scale2..scale2)
        });

        let bias = Array1::zeros(output_dim);

        Self {
            msg_weights,
            update_weights,
            bias,
            input_dim,
            msg_dim,
            output_dim,
            aggregation: AggregationType::Mean,
        }
    }

    /// Forward pass
    pub fn forward(&self, features: &Array2<f64>, adjacency: &Array2<f64>) -> Array2<f64> {
        let n = features.nrows();

        // Compute messages: M = H * W_msg
        let messages = features.dot(&self.msg_weights);

        // Aggregate messages based on adjacency
        let mut aggregated = Array2::zeros((n, self.msg_dim));

        for i in 0..n {
            let mut neighbor_msgs: Vec<Array1<f64>> = Vec::new();

            for j in 0..n {
                if adjacency[[j, i]] > 0.0 {
                    // j is a neighbor of i
                    let weighted_msg = &messages.row(j) * adjacency[[j, i]];
                    neighbor_msgs.push(weighted_msg.to_owned());
                }
            }

            if !neighbor_msgs.is_empty() {
                let agg_msg = match self.aggregation {
                    AggregationType::Sum => {
                        neighbor_msgs.iter().fold(Array1::zeros(self.msg_dim), |acc, m| acc + m)
                    }
                    AggregationType::Mean => {
                        let sum = neighbor_msgs.iter().fold(Array1::zeros(self.msg_dim), |acc, m| acc + m);
                        sum / neighbor_msgs.len() as f64
                    }
                    AggregationType::Max => {
                        let mut max_msg = neighbor_msgs[0].clone();
                        for msg in &neighbor_msgs[1..] {
                            for k in 0..self.msg_dim {
                                max_msg[k] = max_msg[k].max(msg[k]);
                            }
                        }
                        max_msg
                    }
                };

                for k in 0..self.msg_dim {
                    aggregated[[i, k]] = agg_msg[k];
                }
            }
        }

        // Concatenate original features with aggregated messages
        let mut combined = Array2::zeros((n, self.input_dim + self.msg_dim));
        for i in 0..n {
            for j in 0..self.input_dim {
                combined[[i, j]] = features[[i, j]];
            }
            for j in 0..self.msg_dim {
                combined[[i, self.input_dim + j]] = aggregated[[i, j]];
            }
        }

        // Update transformation
        let updated = combined.dot(&self.update_weights);

        // Apply bias and activation
        let mut output = Array2::zeros((n, self.output_dim));
        for i in 0..n {
            for j in 0..self.output_dim {
                output[[i, j]] = relu(updated[[i, j]] + self.bias[j]);
            }
        }

        output
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        self.msg_weights.len() + self.update_weights.len() + self.bias.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gnn_layer() {
        let layer = GNNLayer::new(4, 8);
        assert_eq!(layer.input_dim, 4);
        assert_eq!(layer.output_dim, 8);

        let features = Array2::from_shape_fn((3, 4), |_| rand::random::<f64>());
        let adjacency = Array2::from_shape_fn((3, 3), |(i, j)| {
            if i != j { 0.5 } else { 0.0 }
        });

        let output = layer.forward(&features, &adjacency);
        assert_eq!(output.shape(), &[3, 8]);
    }

    #[test]
    fn test_gcn_layer() {
        let layer = GraphConvLayer::new(4, 8);
        let features = Array2::from_shape_fn((3, 4), |_| rand::random::<f64>());
        let adjacency = Array2::from_shape_fn((3, 3), |(i, j)| {
            if i != j { 0.5 } else { 0.0 }
        });

        let output = layer.forward(&features, &adjacency);
        assert_eq!(output.shape(), &[3, 8]);
    }

    #[test]
    fn test_message_passing_layer() {
        let layer = MessagePassingLayer::new(4, 6, 8);
        let features = Array2::from_shape_fn((3, 4), |_| rand::random::<f64>());
        let adjacency = Array2::from_shape_fn((3, 3), |(i, j)| {
            if i != j { 0.5 } else { 0.0 }
        });

        let output = layer.forward(&features, &adjacency);
        assert_eq!(output.shape(), &[3, 8]);
    }
}
