//! Embedding network for converting market features to embedding vectors

use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationType {
    /// ReLU activation: max(0, x)
    ReLU,
    /// Leaky ReLU: max(0.01*x, x)
    LeakyReLU,
    /// Tanh activation
    Tanh,
    /// Sigmoid activation
    Sigmoid,
    /// No activation (linear)
    Linear,
}

impl Default for ActivationType {
    fn default() -> Self {
        Self::ReLU
    }
}

/// Configuration for embedding network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Input feature dimension
    pub input_dim: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Output embedding dimension
    pub output_dim: usize,
    /// Whether to L2 normalize output embeddings
    pub normalize_embeddings: bool,
    /// Dropout rate (0.0 = no dropout)
    pub dropout_rate: f64,
    /// Activation function type
    pub activation: ActivationType,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            input_dim: 32,
            hidden_dims: vec![64, 128],
            output_dim: 128,
            normalize_embeddings: true,
            dropout_rate: 0.1,
            activation: ActivationType::ReLU,
        }
    }
}

/// Simple feedforward embedding network
///
/// In production, this would be replaced with a proper deep learning framework.
/// This implementation demonstrates the concept using basic linear algebra.
#[derive(Debug, Clone)]
pub struct EmbeddingNetwork {
    config: EmbeddingConfig,
    /// Weight matrices for each layer
    weights: Vec<Array2<f64>>,
    /// Bias vectors for each layer
    biases: Vec<Array1<f64>>,
}

impl EmbeddingNetwork {
    /// Create a new embedding network with random initialization
    pub fn new(config: EmbeddingConfig) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut rng = rand::thread_rng();

        // Build layer dimensions
        let mut dims = vec![config.input_dim];
        dims.extend(&config.hidden_dims);
        dims.push(config.output_dim);

        // Initialize weights using Xavier initialization
        for i in 0..dims.len() - 1 {
            let (in_dim, out_dim) = (dims[i], dims[i + 1]);
            let std = (2.0 / (in_dim + out_dim) as f64).sqrt();
            let normal = Normal::new(0.0, std).unwrap();

            let weight = Array2::from_shape_fn((in_dim, out_dim), |_| {
                rng.sample(normal)
            });
            let bias = Array1::zeros(out_dim);

            weights.push(weight);
            biases.push(bias);
        }

        Self { config, weights, biases }
    }

    /// Apply activation function
    fn apply_activation(&self, x: &mut Array1<f64>) {
        match self.config.activation {
            ActivationType::ReLU => {
                x.mapv_inplace(|v| v.max(0.0));
            }
            ActivationType::LeakyReLU => {
                x.mapv_inplace(|v| if v > 0.0 { v } else { 0.01 * v });
            }
            ActivationType::Tanh => {
                x.mapv_inplace(|v| v.tanh());
            }
            ActivationType::Sigmoid => {
                x.mapv_inplace(|v| 1.0 / (1.0 + (-v).exp()));
            }
            ActivationType::Linear => {
                // No-op
            }
        }
    }

    /// Forward pass through the network
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut x = input.clone();

        // Pass through all layers except the last
        for i in 0..self.weights.len() - 1 {
            // Linear transformation
            x = self.weights[i].t().dot(&x) + &self.biases[i];
            // Apply activation
            self.apply_activation(&mut x);
        }

        // Last layer (no activation for embedding)
        let last_idx = self.weights.len() - 1;
        x = self.weights[last_idx].t().dot(&x) + &self.biases[last_idx];

        // Optional L2 normalization
        if self.config.normalize_embeddings {
            let norm = x.dot(&x).sqrt();
            if norm > 1e-8 {
                x /= norm;
            }
        }

        x
    }

    /// Forward pass for batch of inputs
    pub fn forward_batch(&self, inputs: &Array2<f64>) -> Array2<f64> {
        let n_samples = inputs.nrows();
        let mut outputs = Array2::zeros((n_samples, self.config.output_dim));

        for i in 0..n_samples {
            let input = inputs.row(i).to_owned();
            let output = self.forward(&input);
            outputs.row_mut(i).assign(&output);
        }

        outputs
    }

    /// Get the output embedding dimension
    pub fn output_dim(&self) -> usize {
        self.config.output_dim
    }

    /// Get the embedding dimension (alias for output_dim)
    pub fn embedding_dim(&self) -> usize {
        self.config.output_dim
    }

    /// Get the input dimension
    pub fn input_dim(&self) -> usize {
        self.config.input_dim
    }

    /// Scale all weights by a factor (for weight decay)
    pub fn scale_weights(&mut self, factor: f64) {
        for weight in &mut self.weights {
            weight.mapv_inplace(|w| w * factor);
        }
    }

    /// Get reference to config
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_network_creation() {
        let config = EmbeddingConfig {
            input_dim: 16,
            hidden_dims: vec![32, 64],
            output_dim: 32,
            normalize_embeddings: true,
            dropout_rate: 0.0,
            activation: ActivationType::ReLU,
        };
        let network = EmbeddingNetwork::new(config);

        assert_eq!(network.weights.len(), 3); // 3 layers
        assert_eq!(network.input_dim(), 16);
        assert_eq!(network.output_dim(), 32);
        assert_eq!(network.embedding_dim(), 32);
    }

    #[test]
    fn test_forward_pass() {
        let config = EmbeddingConfig {
            input_dim: 8,
            hidden_dims: vec![16],
            output_dim: 4,
            normalize_embeddings: false,
            dropout_rate: 0.0,
            activation: ActivationType::ReLU,
        };
        let network = EmbeddingNetwork::new(config);

        let input = Array1::from_vec(vec![1.0; 8]);
        let output = network.forward(&input);

        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_normalized_output() {
        let config = EmbeddingConfig {
            input_dim: 8,
            hidden_dims: vec![16],
            output_dim: 4,
            normalize_embeddings: true,
            dropout_rate: 0.0,
            activation: ActivationType::ReLU,
        };
        let network = EmbeddingNetwork::new(config);

        let input = Array1::from_vec(vec![1.0; 8]);
        let output = network.forward(&input);

        // Check L2 norm is approximately 1
        let norm = output.dot(&output).sqrt();
        assert!((norm - 1.0).abs() < 0.01 || norm < 1e-6);
    }

    #[test]
    fn test_batch_forward() {
        let config = EmbeddingConfig::default();
        let network = EmbeddingNetwork::new(config.clone());

        let batch_size = 5;
        let inputs = Array2::from_shape_fn((batch_size, config.input_dim), |_| 0.5);
        let outputs = network.forward_batch(&inputs);

        assert_eq!(outputs.nrows(), batch_size);
        assert_eq!(outputs.ncols(), config.output_dim);
    }

    #[test]
    fn test_scale_weights() {
        let config = EmbeddingConfig {
            input_dim: 4,
            hidden_dims: vec![8],
            output_dim: 4,
            normalize_embeddings: false,
            dropout_rate: 0.0,
            activation: ActivationType::ReLU,
        };
        let mut network = EmbeddingNetwork::new(config);

        let original_sum: f64 = network.weights.iter().map(|w| w.sum()).sum();
        network.scale_weights(0.5);
        let scaled_sum: f64 = network.weights.iter().map(|w| w.sum()).sum();

        assert!((scaled_sum - original_sum * 0.5).abs() < 1e-10);
    }
}
