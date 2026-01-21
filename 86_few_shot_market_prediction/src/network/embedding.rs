//! Embedding network for few-shot learning
//!
//! Supports both forward pass and parameter adaptation for MAML-style training.

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
    /// GELU (Gaussian Error Linear Unit)
    GELU,
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
    /// Whether to use batch normalization
    pub use_batch_norm: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            input_dim: 20,
            hidden_dims: vec![128, 64],
            output_dim: 32,
            normalize_embeddings: true,
            dropout_rate: 0.1,
            activation: ActivationType::ReLU,
            use_batch_norm: true,
        }
    }
}

impl EmbeddingConfig {
    /// Create a new config with specified dimensions
    pub fn new(input_dim: usize, hidden_dims: Vec<usize>, output_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dims,
            output_dim,
            ..Default::default()
        }
    }

    /// Builder pattern: set input dimension
    pub fn with_input_dim(mut self, dim: usize) -> Self {
        self.input_dim = dim;
        self
    }

    /// Builder pattern: set output dimension
    pub fn with_output_dim(mut self, dim: usize) -> Self {
        self.output_dim = dim;
        self
    }

    /// Builder pattern: set activation
    pub fn with_activation(mut self, activation: ActivationType) -> Self {
        self.activation = activation;
        self
    }
}

/// Network parameters that can be adapted during MAML training
#[derive(Debug, Clone)]
pub struct NetworkParams {
    /// Weight matrices for each layer
    pub weights: Vec<Array2<f64>>,
    /// Bias vectors for each layer
    pub biases: Vec<Array1<f64>>,
}

impl NetworkParams {
    /// Create new random parameters
    pub fn random(layer_dims: &[usize]) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut rng = rand::thread_rng();

        for i in 0..layer_dims.len() - 1 {
            let (in_dim, out_dim) = (layer_dims[i], layer_dims[i + 1]);
            let std = (2.0 / (in_dim + out_dim) as f64).sqrt();
            let normal = Normal::new(0.0, std).unwrap();

            let weight = Array2::from_shape_fn((in_dim, out_dim), |_| rng.sample(normal));
            let bias = Array1::zeros(out_dim);

            weights.push(weight);
            biases.push(bias);
        }

        Self { weights, biases }
    }

    /// Clone parameters
    pub fn clone_params(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            biases: self.biases.clone(),
        }
    }

    /// Apply gradient update with given learning rate
    pub fn apply_gradients(&mut self, weight_grads: &[Array2<f64>], bias_grads: &[Array1<f64>], lr: f64) {
        for (i, (wg, bg)) in weight_grads.iter().zip(bias_grads.iter()).enumerate() {
            self.weights[i] = &self.weights[i] - &(wg * lr);
            self.biases[i] = &self.biases[i] - &(bg * lr);
        }
    }

    /// Scale all weights by a factor (for weight decay)
    pub fn scale_weights(&mut self, factor: f64) {
        for weight in &mut self.weights {
            weight.mapv_inplace(|w| w * factor);
        }
    }
}

/// Embedding network for converting market features to embedding vectors
///
/// Supports MAML-style training by allowing parameter injection during forward pass.
#[derive(Debug, Clone)]
pub struct EmbeddingNetwork {
    config: EmbeddingConfig,
    /// Default parameters (can be overridden during forward pass)
    params: NetworkParams,
    /// Layer dimensions for easy access
    layer_dims: Vec<usize>,
}

impl EmbeddingNetwork {
    /// Create a new embedding network with random initialization
    pub fn new(config: EmbeddingConfig) -> Self {
        // Build layer dimensions
        let mut layer_dims = vec![config.input_dim];
        layer_dims.extend(&config.hidden_dims);
        layer_dims.push(config.output_dim);

        let params = NetworkParams::random(&layer_dims);

        Self {
            config,
            params,
            layer_dims,
        }
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
            ActivationType::GELU => {
                // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                x.mapv_inplace(|v| {
                    let inner = 0.7978845608 * (v + 0.044715 * v.powi(3));
                    0.5 * v * (1.0 + inner.tanh())
                });
            }
            ActivationType::Linear => {
                // No-op
            }
        }
    }

    /// Forward pass through the network with optional custom parameters
    ///
    /// # Arguments
    /// * `input` - Input features
    /// * `params` - Optional custom parameters (for MAML adaptation)
    pub fn forward(&self, input: &Array1<f64>, params: Option<&NetworkParams>) -> Array1<f64> {
        let params = params.unwrap_or(&self.params);
        let mut x = input.clone();

        // Pass through all layers except the last
        for i in 0..params.weights.len() - 1 {
            // Linear transformation
            x = params.weights[i].t().dot(&x) + &params.biases[i];
            // Apply activation
            self.apply_activation(&mut x);
        }

        // Last layer (no activation for embedding)
        let last_idx = params.weights.len() - 1;
        x = params.weights[last_idx].t().dot(&x) + &params.biases[last_idx];

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
    pub fn forward_batch(&self, inputs: &Array2<f64>, params: Option<&NetworkParams>) -> Array2<f64> {
        let n_samples = inputs.nrows();
        let mut outputs = Array2::zeros((n_samples, self.config.output_dim));

        for i in 0..n_samples {
            let input = inputs.row(i).to_owned();
            let output = self.forward(&input, params);
            outputs.row_mut(i).assign(&output);
        }

        outputs
    }

    /// Get current parameters
    pub fn get_params(&self) -> &NetworkParams {
        &self.params
    }

    /// Get mutable reference to parameters
    pub fn get_params_mut(&mut self) -> &mut NetworkParams {
        &mut self.params
    }

    /// Set parameters
    pub fn set_params(&mut self, params: NetworkParams) {
        self.params = params;
    }

    /// Clone current parameters
    pub fn clone_params(&self) -> NetworkParams {
        self.params.clone_params()
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

    /// Get layer dimensions
    pub fn layer_dims(&self) -> &[usize] {
        &self.layer_dims
    }

    /// Get reference to config
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let config = EmbeddingConfig {
            input_dim: 16,
            hidden_dims: vec![32, 64],
            output_dim: 32,
            normalize_embeddings: true,
            dropout_rate: 0.0,
            activation: ActivationType::ReLU,
            use_batch_norm: false,
        };
        let network = EmbeddingNetwork::new(config);

        assert_eq!(network.params.weights.len(), 3);
        assert_eq!(network.input_dim(), 16);
        assert_eq!(network.output_dim(), 32);
    }

    #[test]
    fn test_forward_pass() {
        let config = EmbeddingConfig::new(8, vec![16], 4);
        let network = EmbeddingNetwork::new(config);

        let input = Array1::from_vec(vec![1.0; 8]);
        let output = network.forward(&input, None);

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
            use_batch_norm: false,
        };
        let network = EmbeddingNetwork::new(config);

        let input = Array1::from_vec(vec![1.0; 8]);
        let output = network.forward(&input, None);

        let norm = output.dot(&output).sqrt();
        assert!((norm - 1.0).abs() < 0.01 || norm < 1e-6);
    }

    #[test]
    fn test_forward_with_custom_params() {
        // Disable normalization so scaling weights produces different outputs
        let config = EmbeddingConfig {
            input_dim: 4,
            hidden_dims: vec![8],
            output_dim: 4,
            normalize_embeddings: false, // Disable normalization for this test
            dropout_rate: 0.0,
            activation: ActivationType::ReLU,
            use_batch_norm: false,
        };
        let network = EmbeddingNetwork::new(config);

        // Clone and modify parameters by adding a bias offset (not just scaling)
        let mut custom_params = network.clone_params();
        for bias in &mut custom_params.biases {
            bias.mapv_inplace(|b| b + 0.5);
        }

        let input = Array1::from_vec(vec![0.5; 4]);

        let output_default = network.forward(&input, None);
        let output_custom = network.forward(&input, Some(&custom_params));

        // Outputs should be different due to bias change
        let diff: f64 = (&output_default - &output_custom).mapv(|x| x.abs()).sum();
        assert!(diff > 0.0, "Expected diff > 0.0, got {}", diff);
    }

    #[test]
    fn test_batch_forward() {
        let config = EmbeddingConfig::default();
        let network = EmbeddingNetwork::new(config.clone());

        let batch_size = 5;
        let inputs = Array2::from_shape_fn((batch_size, config.input_dim), |_| 0.5);
        let outputs = network.forward_batch(&inputs, None);

        assert_eq!(outputs.nrows(), batch_size);
        assert_eq!(outputs.ncols(), config.output_dim);
    }
}
