//! Mixture of Experts (MoE) encoder for specialized market pattern processing

use super::SharedEncoder;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde::{Deserialize, Serialize};

/// MoE encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoEConfig {
    /// Input feature dimension
    pub input_dim: usize,
    /// Output embedding dimension
    pub embedding_dim: usize,
    /// Number of expert networks
    pub num_experts: usize,
    /// Hidden dimension for each expert
    pub expert_dim: usize,
    /// Number of experts to route to (top-k)
    pub top_k: usize,
    /// Dropout rate
    pub dropout: f64,
}

impl Default for MoEConfig {
    fn default() -> Self {
        Self {
            input_dim: 20,
            embedding_dim: 64,
            num_experts: 4,
            expert_dim: 64,
            top_k: 2,
            dropout: 0.1,
        }
    }
}

/// Single expert network (MLP)
struct Expert {
    w1: Array2<f64>,
    w2: Array2<f64>,
}

impl Expert {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let scale1 = (2.0 / (input_dim + hidden_dim) as f64).sqrt();
        let scale2 = (2.0 / (hidden_dim + output_dim) as f64).sqrt();

        Self {
            w1: Array2::random((input_dim, hidden_dim), Uniform::new(-scale1, scale1)),
            w2: Array2::random((hidden_dim, output_dim), Uniform::new(-scale2, scale2)),
        }
    }

    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        // Two-layer MLP with ReLU
        let hidden = input.dot(&self.w1).mapv(|x| x.max(0.0));
        hidden.dot(&self.w2)
    }
}

/// Gating network for expert routing
struct GatingNetwork {
    weights: Array2<f64>,
}

impl GatingNetwork {
    fn new(input_dim: usize, num_experts: usize) -> Self {
        let scale = (2.0 / (input_dim + num_experts) as f64).sqrt();
        Self {
            weights: Array2::random((input_dim, num_experts), Uniform::new(-scale, scale)),
        }
    }

    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let logits = input.dot(&self.weights);
        softmax(&logits)
    }
}

/// Softmax function
fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max_val = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_x = x.mapv(|v| (v - max_val).exp());
    let sum = exp_x.sum();
    exp_x / sum
}

/// Get top-k indices and values
fn top_k(arr: &Array1<f64>, k: usize) -> (Vec<usize>, Vec<f64>) {
    let mut indexed: Vec<(usize, f64)> = arr.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_indices: Vec<usize> = indexed.iter().take(k).map(|(i, _)| *i).collect();
    let top_values: Vec<f64> = indexed.iter().take(k).map(|(_, v)| *v).collect();

    (top_indices, top_values)
}

/// Mixture of Experts encoder
pub struct MoEEncoder {
    config: MoEConfig,
    input_projection: Array2<f64>,
    experts: Vec<Expert>,
    gating: GatingNetwork,
    output_projection: Array2<f64>,
}

impl MoEEncoder {
    /// Create a new MoE encoder
    pub fn new(config: MoEConfig) -> Self {
        let scale_in = (2.0 / (config.input_dim + config.expert_dim) as f64).sqrt();
        let input_projection = Array2::random(
            (config.input_dim, config.expert_dim),
            Uniform::new(-scale_in, scale_in),
        );

        let experts: Vec<_> = (0..config.num_experts)
            .map(|_| Expert::new(config.expert_dim, config.expert_dim * 2, config.expert_dim))
            .collect();

        let gating = GatingNetwork::new(config.expert_dim, config.num_experts);

        let scale_out = (2.0 / (config.expert_dim + config.embedding_dim) as f64).sqrt();
        let output_projection = Array2::random(
            (config.expert_dim, config.embedding_dim),
            Uniform::new(-scale_out, scale_out),
        );

        Self {
            config,
            input_projection,
            experts,
            gating,
            output_projection,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &MoEConfig {
        &self.config
    }

    /// Get expert activations (for analysis/debugging)
    pub fn get_expert_weights(&self, input: &Array1<f64>) -> Array1<f64> {
        let projected = input.dot(&self.input_projection);
        self.gating.forward(&projected)
    }
}

impl SharedEncoder for MoEEncoder {
    fn encode(&self, input: &Array1<f64>) -> Array1<f64> {
        // Project input
        let projected = input.dot(&self.input_projection);

        // Get gating weights
        let gate_weights = self.gating.forward(&projected);

        // Select top-k experts
        let (top_indices, top_weights) = top_k(&gate_weights, self.config.top_k);

        // Renormalize top-k weights
        let weight_sum: f64 = top_weights.iter().sum();
        let normalized_weights: Vec<f64> = top_weights.iter().map(|w| w / weight_sum).collect();

        // Compute weighted sum of expert outputs
        let mut output = Array1::zeros(self.config.expert_dim);
        for (idx, weight) in top_indices.iter().zip(normalized_weights.iter()) {
            let expert_out = self.experts[*idx].forward(&projected);
            output = output + &(expert_out * *weight);
        }

        // Final projection
        output.dot(&self.output_projection)
    }

    fn encode_batch(&self, inputs: &Array2<f64>) -> Array2<f64> {
        let mut outputs = Vec::with_capacity(inputs.nrows());
        for row in inputs.axis_iter(Axis(0)) {
            let embedding = self.encode(&row.to_owned());
            outputs.push(embedding);
        }

        let flat: Vec<f64> = outputs.iter().flat_map(|e| e.to_vec()).collect();
        Array2::from_shape_vec((inputs.nrows(), self.config.embedding_dim), flat)
            .expect("Shape mismatch in encode_batch")
    }

    fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    fn parameters(&self) -> Vec<Array2<f64>> {
        vec![
            self.input_projection.clone(),
            self.output_projection.clone(),
        ]
    }

    fn update_parameters(&mut self, gradients: &[Array2<f64>], learning_rate: f64) {
        if gradients.len() >= 2 {
            self.input_projection = &self.input_projection - &(&gradients[0] * learning_rate);
            self.output_projection = &self.output_projection - &(&gradients[1] * learning_rate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_moe_encoder() {
        let config = MoEConfig {
            input_dim: 20,
            embedding_dim: 32,
            num_experts: 4,
            expert_dim: 16,
            top_k: 2,
            dropout: 0.1,
        };

        let encoder = MoEEncoder::new(config);
        let input = Array::random(20, Uniform::new(-1.0, 1.0));
        let output = encoder.encode(&input);

        assert_eq!(output.len(), 32);
    }

    #[test]
    fn test_gating() {
        let config = MoEConfig::default();
        let encoder = MoEEncoder::new(config.clone());
        let input = Array::random(config.input_dim, Uniform::new(-1.0, 1.0));

        let weights = encoder.get_expert_weights(&input);

        // Weights should sum to 1 (softmax)
        let sum: f64 = weights.sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_k() {
        let arr = Array1::from_vec(vec![0.1, 0.4, 0.2, 0.3]);
        let (indices, values) = top_k(&arr, 2);

        assert_eq!(indices, vec![1, 3]);
        assert!((values[0] - 0.4).abs() < 1e-10);
        assert!((values[1] - 0.3).abs() < 1e-10);
    }
}
