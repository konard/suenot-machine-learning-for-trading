//! Metapath encoder for heterogeneous graphs

use ndarray::Array1;
use std::collections::HashMap;

use super::layers::LinearLayer;
use super::attention::MultiHeadAttention;
use crate::graph::{Metapath, MetapathInstance};

/// Metapath encoder for learning metapath-based representations
#[derive(Debug)]
pub struct MetapathEncoder {
    /// Hidden dimension
    hidden_dim: usize,
    /// Number of samples per metapath
    num_samples: usize,
    /// LSTM-like cell for sequence encoding
    path_encoder: PathEncoder,
    /// Attention for aggregating path instances
    path_attention: MultiHeadAttention,
    /// Metapath importance weights (learned)
    metapath_weights: HashMap<String, f64>,
}

impl MetapathEncoder {
    /// Create a new metapath encoder
    pub fn new(hidden_dim: usize, num_samples: usize) -> Self {
        Self {
            hidden_dim,
            num_samples,
            path_encoder: PathEncoder::new(hidden_dim),
            path_attention: MultiHeadAttention::new(hidden_dim, 4),
            metapath_weights: HashMap::new(),
        }
    }

    /// Encode a metapath instance
    pub fn encode_instance(
        &self,
        instance: &MetapathInstance,
        node_embeddings: &HashMap<String, Array1<f64>>,
    ) -> Array1<f64> {
        // Get embeddings for nodes in the path
        let path_embeddings: Vec<Array1<f64>> = instance.node_ids
            .iter()
            .filter_map(|id| node_embeddings.get(id).cloned())
            .collect();

        if path_embeddings.is_empty() {
            return Array1::zeros(self.hidden_dim);
        }

        // Encode the path sequence
        self.path_encoder.encode(&path_embeddings) * instance.weight
    }

    /// Encode multiple instances of the same metapath
    pub fn encode_metapath(
        &self,
        instances: &[MetapathInstance],
        node_embeddings: &HashMap<String, Array1<f64>>,
    ) -> Array1<f64> {
        if instances.is_empty() {
            return Array1::zeros(self.hidden_dim);
        }

        // Encode each instance
        let instance_embeddings: Vec<Array1<f64>> = instances
            .iter()
            .map(|inst| self.encode_instance(inst, node_embeddings))
            .collect();

        // Aggregate using attention
        self.path_attention.aggregate(&instance_embeddings)
    }

    /// Get metapath importance weight
    pub fn get_weight(&self, metapath_name: &str) -> f64 {
        *self.metapath_weights.get(metapath_name).unwrap_or(&1.0)
    }

    /// Set metapath importance weight
    pub fn set_weight(&mut self, metapath_name: &str, weight: f64) {
        self.metapath_weights.insert(metapath_name.to_string(), weight);
    }
}

/// Path encoder using GRU-like recurrent processing
#[derive(Debug)]
pub struct PathEncoder {
    /// Hidden dimension
    hidden_dim: usize,
    /// Update gate
    update_gate: LinearLayer,
    /// Reset gate
    reset_gate: LinearLayer,
    /// Candidate hidden state
    candidate: LinearLayer,
}

impl PathEncoder {
    /// Create a new path encoder
    pub fn new(hidden_dim: usize) -> Self {
        // GRU-like architecture: takes concatenation of input and hidden
        let combined_dim = hidden_dim * 2;

        Self {
            hidden_dim,
            update_gate: LinearLayer::new(combined_dim, hidden_dim),
            reset_gate: LinearLayer::new(combined_dim, hidden_dim),
            candidate: LinearLayer::new(combined_dim, hidden_dim),
        }
    }

    /// Encode a sequence of embeddings
    pub fn encode(&self, sequence: &[Array1<f64>]) -> Array1<f64> {
        if sequence.is_empty() {
            return Array1::zeros(self.hidden_dim);
        }

        let mut hidden = Array1::zeros(self.hidden_dim);

        for embedding in sequence {
            hidden = self.step(embedding, &hidden);
        }

        hidden
    }

    /// Single GRU step
    fn step(&self, input: &Array1<f64>, hidden: &Array1<f64>) -> Array1<f64> {
        // Resize input if necessary
        let mut resized_input = Array1::zeros(self.hidden_dim);
        for (i, &val) in input.iter().enumerate() {
            if i < self.hidden_dim {
                resized_input[i] = val;
            }
        }

        // Concatenate input and hidden
        let mut combined = Array1::zeros(self.hidden_dim * 2);
        for (i, &val) in resized_input.iter().enumerate() {
            combined[i] = val;
        }
        for (i, &val) in hidden.iter().enumerate() {
            combined[self.hidden_dim + i] = val;
        }

        // Update gate: z = sigmoid(W_z * [x, h])
        let z = self.update_gate.forward(&combined).mapv(|x| 1.0 / (1.0 + (-x).exp()));

        // Reset gate: r = sigmoid(W_r * [x, h])
        let r = self.reset_gate.forward(&combined).mapv(|x| 1.0 / (1.0 + (-x).exp()));

        // Candidate hidden: h_tilde = tanh(W_c * [x, r * h])
        let mut reset_combined = Array1::zeros(self.hidden_dim * 2);
        for (i, &val) in resized_input.iter().enumerate() {
            reset_combined[i] = val;
        }
        for (i, (&r_i, &h_i)) in r.iter().zip(hidden.iter()).enumerate() {
            reset_combined[self.hidden_dim + i] = r_i * h_i;
        }
        let h_tilde = self.candidate.forward(&reset_combined).mapv(|x| x.tanh());

        // New hidden: h_new = (1 - z) * h + z * h_tilde
        let mut h_new = Array1::zeros(self.hidden_dim);
        for i in 0..self.hidden_dim {
            h_new[i] = (1.0 - z[i]) * hidden[i] + z[i] * h_tilde[i];
        }

        h_new
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_encoder() {
        let encoder = PathEncoder::new(16);
        let sequence = vec![
            Array1::ones(16),
            Array1::ones(16) * 0.5,
            Array1::ones(16) * 2.0,
        ];

        let output = encoder.encode(&sequence);
        assert_eq!(output.len(), 16);
    }

    #[test]
    fn test_metapath_encoder() {
        use crate::graph::{Metapath, NodeType, EdgeType};

        let encoder = MetapathEncoder::new(16, 10);

        let metapath = Metapath::new(
            "test",
            "Test metapath",
            vec![
                (NodeType::Asset, Some(EdgeType::Correlation)),
                (NodeType::Asset, None),
            ],
        );

        let instance = MetapathInstance::new(
            metapath,
            vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()],
        );

        let mut node_embeddings = HashMap::new();
        node_embeddings.insert("BTCUSDT".to_string(), Array1::ones(16));
        node_embeddings.insert("ETHUSDT".to_string(), Array1::ones(16) * 0.5);

        let output = encoder.encode_instance(&instance, &node_embeddings);
        assert_eq!(output.len(), 16);
    }
}
