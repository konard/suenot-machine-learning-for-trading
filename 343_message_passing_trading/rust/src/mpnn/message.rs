//! Message functions for MPNN.
//!
//! Message functions compute the information sent from one node to another.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Types of message functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageType {
    /// Simple: M(h_v, h_w, e_vw) = h_w
    Simple,
    /// Edge-weighted: M(h_v, h_w, e_vw) = e_vw * h_w
    EdgeWeighted,
    /// Concatenated: M(h_v, h_w, e_vw) = [h_v || h_w]
    Concatenated,
    /// MLP-based: M(h_v, h_w, e_vw) = MLP([h_v || h_w || e_vw])
    MLP,
}

/// Message function that computes messages between nodes.
pub struct MessageFunction {
    /// Type of message function
    pub message_type: MessageType,
    /// Optional weight matrix for transformation
    pub weights: Option<Array2<f64>>,
    /// Optional MLP weights for MLP message type
    pub mlp_weights: Option<Vec<Array2<f64>>>,
}

impl MessageFunction {
    /// Create a simple message function.
    pub fn simple() -> Self {
        Self {
            message_type: MessageType::Simple,
            weights: None,
            mlp_weights: None,
        }
    }

    /// Create an edge-weighted message function.
    pub fn edge_weighted() -> Self {
        Self {
            message_type: MessageType::EdgeWeighted,
            weights: None,
            mlp_weights: None,
        }
    }

    /// Create a concatenated message function with learned transformation.
    pub fn concatenated(input_dim: usize, output_dim: usize) -> Self {
        use rand::Rng;
        use rand_distr::{Distribution, Normal};

        let mut rng = rand::thread_rng();
        let scale = (2.0 / (2 * input_dim + output_dim) as f64).sqrt();
        let normal = Normal::new(0.0, scale).unwrap();

        let weights = Array2::from_shape_fn((2 * input_dim, output_dim), |_| {
            normal.sample(&mut rng)
        });

        Self {
            message_type: MessageType::Concatenated,
            weights: Some(weights),
            mlp_weights: None,
        }
    }

    /// Create an MLP-based message function.
    pub fn mlp(input_dim: usize, hidden_dim: usize, output_dim: usize, edge_dim: usize) -> Self {
        use rand::Rng;
        use rand_distr::{Distribution, Normal};

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        let total_input = 2 * input_dim + edge_dim;
        let mlp_weights = vec![
            Array2::from_shape_fn((total_input, hidden_dim), |_| normal.sample(&mut rng)),
            Array2::from_shape_fn((hidden_dim, output_dim), |_| normal.sample(&mut rng)),
        ];

        Self {
            message_type: MessageType::MLP,
            weights: None,
            mlp_weights: Some(mlp_weights),
        }
    }

    /// Compute message from source to target node.
    pub fn compute(
        &self,
        source_features: &Array1<f64>,
        target_features: &Array1<f64>,
        edge_weight: f64,
        edge_features: Option<&Array1<f64>>,
    ) -> Array1<f64> {
        match self.message_type {
            MessageType::Simple => source_features.clone(),

            MessageType::EdgeWeighted => source_features.mapv(|x| x * edge_weight),

            MessageType::Concatenated => {
                if let Some(w) = &self.weights {
                    // Concatenate source and target features
                    let mut concat = Vec::with_capacity(source_features.len() * 2);
                    concat.extend(source_features.iter());
                    concat.extend(target_features.iter());
                    let concat = Array1::from_vec(concat);

                    // Transform
                    let result = concat.dot(w);

                    // ReLU activation
                    result.mapv(|x| x.max(0.0))
                } else {
                    source_features.clone()
                }
            }

            MessageType::MLP => {
                if let Some(mlp) = &self.mlp_weights {
                    // Concatenate all features
                    let mut concat = Vec::with_capacity(
                        source_features.len() * 2 + edge_features.map(|e| e.len()).unwrap_or(0),
                    );
                    concat.extend(source_features.iter());
                    concat.extend(target_features.iter());
                    if let Some(ef) = edge_features {
                        concat.extend(ef.iter());
                    }
                    let concat = Array1::from_vec(concat);

                    // Forward through MLP
                    let mut h = concat;
                    for (i, w) in mlp.iter().enumerate() {
                        if h.len() != w.nrows() {
                            // Dimension mismatch, return zeros
                            return Array1::zeros(w.ncols());
                        }
                        h = h.dot(w);
                        // Apply ReLU except for last layer
                        if i < mlp.len() - 1 {
                            h = h.mapv(|x| x.max(0.0));
                        }
                    }
                    h
                } else {
                    source_features.clone()
                }
            }
        }
    }

    /// Compute all messages for a node from its neighbors.
    pub fn compute_all_messages(
        &self,
        target_id: usize,
        node_features: &Array2<f64>,
        neighbors: &[(usize, f64)],
        edge_features: Option<&[Array1<f64>]>,
    ) -> Vec<Array1<f64>> {
        let target_features = node_features.row(target_id).to_owned();

        neighbors
            .iter()
            .enumerate()
            .map(|(i, &(source_id, weight))| {
                let source_features = node_features.row(source_id).to_owned();
                let edge_feat = edge_features.and_then(|ef| ef.get(i));
                self.compute(&source_features, &target_features, weight, edge_feat)
            })
            .collect()
    }
}

/// Trait for custom message functions.
pub trait Message {
    /// Compute the message from source to target.
    fn message(
        &self,
        source: &Array1<f64>,
        target: &Array1<f64>,
        edge: Option<&Array1<f64>>,
    ) -> Array1<f64>;
}

/// A simple linear message function.
pub struct LinearMessage {
    /// Weight matrix
    pub weights: Array2<f64>,
}

impl LinearMessage {
    /// Create a new linear message function.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, (2.0 / input_dim as f64).sqrt()).unwrap();

        Self {
            weights: Array2::from_shape_fn((input_dim, output_dim), |_| normal.sample(&mut rng)),
        }
    }
}

impl Message for LinearMessage {
    fn message(
        &self,
        source: &Array1<f64>,
        _target: &Array1<f64>,
        _edge: Option<&Array1<f64>>,
    ) -> Array1<f64> {
        source.dot(&self.weights)
    }
}

/// Gated message function with edge gating.
pub struct GatedMessage {
    /// Gate network weights
    pub gate_weights: Array2<f64>,
    /// Transform weights
    pub transform_weights: Array2<f64>,
}

impl GatedMessage {
    /// Create a new gated message function.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        Self {
            gate_weights: Array2::from_shape_fn((input_dim * 2, output_dim), |_| {
                normal.sample(&mut rng)
            }),
            transform_weights: Array2::from_shape_fn((input_dim, output_dim), |_| {
                normal.sample(&mut rng)
            }),
        }
    }
}

impl Message for GatedMessage {
    fn message(
        &self,
        source: &Array1<f64>,
        target: &Array1<f64>,
        _edge: Option<&Array1<f64>>,
    ) -> Array1<f64> {
        // Compute gate
        let mut concat = Vec::with_capacity(source.len() * 2);
        concat.extend(source.iter());
        concat.extend(target.iter());
        let concat = Array1::from_vec(concat);

        let gate_logits = concat.dot(&self.gate_weights);
        let gate: Array1<f64> = gate_logits.mapv(|x| 1.0 / (1.0 + (-x).exp())); // Sigmoid

        // Compute transformed message
        let transformed = source.dot(&self.transform_weights);

        // Apply gate
        &gate * &transformed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_simple_message() {
        let msg_fn = MessageFunction::simple();
        let source = array![1.0, 2.0, 3.0];
        let target = array![0.5, 0.5, 0.5];

        let message = msg_fn.compute(&source, &target, 1.0, None);
        assert_eq!(message, source);
    }

    #[test]
    fn test_edge_weighted_message() {
        let msg_fn = MessageFunction::edge_weighted();
        let source = array![1.0, 2.0, 3.0];
        let target = array![0.5, 0.5, 0.5];

        let message = msg_fn.compute(&source, &target, 0.5, None);
        assert_eq!(message, array![0.5, 1.0, 1.5]);
    }

    #[test]
    fn test_concatenated_message() {
        let msg_fn = MessageFunction::concatenated(3, 4);
        let source = array![1.0, 2.0, 3.0];
        let target = array![0.5, 0.5, 0.5];

        let message = msg_fn.compute(&source, &target, 1.0, None);
        assert_eq!(message.len(), 4);
    }

    #[test]
    fn test_mlp_message() {
        let msg_fn = MessageFunction::mlp(3, 8, 4, 2);
        let source = array![1.0, 2.0, 3.0];
        let target = array![0.5, 0.5, 0.5];
        let edge = array![0.8, 0.2];

        let message = msg_fn.compute(&source, &target, 1.0, Some(&edge));
        assert_eq!(message.len(), 4);
    }
}
