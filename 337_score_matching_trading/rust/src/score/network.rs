//! # Score Network Implementation
//!
//! Neural network architecture for estimating the score function s(x) = ∇_x log p(x).
//!
//! Implements a ResNet-style MLP with skip connections.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

/// Activation function types
#[derive(Debug, Clone, Copy)]
pub enum ActivationFn {
    /// Gaussian Error Linear Unit
    GELU,
    /// Rectified Linear Unit
    ReLU,
    /// Leaky ReLU with given negative slope
    LeakyReLU(f64),
    /// Hyperbolic tangent
    Tanh,
    /// Sigmoid
    Sigmoid,
    /// Identity (no activation)
    Identity,
}

impl ActivationFn {
    /// Apply activation function
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            ActivationFn::GELU => {
                // GELU(x) = x * Φ(x) where Φ is standard normal CDF
                // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                let sqrt_2_pi = (2.0 / PI).sqrt();
                0.5 * x * (1.0 + (sqrt_2_pi * (x + 0.044715 * x.powi(3))).tanh())
            }
            ActivationFn::ReLU => x.max(0.0),
            ActivationFn::LeakyReLU(alpha) => if x > 0.0 { x } else { alpha * x },
            ActivationFn::Tanh => x.tanh(),
            ActivationFn::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFn::Identity => x,
        }
    }

    /// Compute derivative of activation function
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFn::GELU => {
                let sqrt_2_pi = (2.0 / PI).sqrt();
                let inner = sqrt_2_pi * (x + 0.044715 * x.powi(3));
                let tanh_inner = inner.tanh();
                let sech2 = 1.0 - tanh_inner.powi(2);
                let inner_deriv = sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x.powi(2));
                0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * inner_deriv
            }
            ActivationFn::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            ActivationFn::LeakyReLU(alpha) => if x > 0.0 { 1.0 } else { *alpha },
            ActivationFn::Tanh => 1.0 - x.tanh().powi(2),
            ActivationFn::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            ActivationFn::Identity => 1.0,
        }
    }

    /// Apply activation to array
    pub fn apply_array(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| self.apply(v))
    }
}

/// Single layer in the neural network
#[derive(Debug, Clone)]
pub struct Layer {
    /// Weight matrix (output_dim x input_dim)
    pub weights: Array2<f64>,
    /// Bias vector (output_dim)
    pub bias: Array1<f64>,
    /// Activation function
    pub activation: ActivationFn,
}

impl Layer {
    /// Create a new layer with random initialization
    pub fn new(input_dim: usize, output_dim: usize, activation: ActivationFn) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier/Glorot initialization
        let std = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let weights = Array2::from_shape_fn((output_dim, input_dim), |_| normal.sample(&mut rng));
        let bias = Array1::zeros(output_dim);

        Self {
            weights,
            bias,
            activation,
        }
    }

    /// Forward pass through the layer
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let z = self.weights.dot(input) + &self.bias;
        self.activation.apply_array(&z)
    }

    /// Forward pass with pre-activation output
    pub fn forward_with_preactivation(&self, input: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let z = self.weights.dot(input) + &self.bias;
        let a = self.activation.apply_array(&z);
        (z, a)
    }
}

/// Residual block for skip connections
#[derive(Debug, Clone)]
pub struct ResidualBlock {
    /// First layer
    layer1: Layer,
    /// Second layer
    layer2: Layer,
    /// Dropout probability (stored but not used in inference)
    dropout: f64,
}

impl ResidualBlock {
    /// Create a new residual block
    pub fn new(dim: usize, hidden_mult: usize, dropout: f64) -> Self {
        let hidden_dim = dim * hidden_mult;
        Self {
            layer1: Layer::new(dim, hidden_dim, ActivationFn::GELU),
            layer2: Layer::new(hidden_dim, dim, ActivationFn::Identity),
            dropout,
        }
    }

    /// Forward pass with residual connection
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let h = self.layer1.forward(input);
        let out = self.layer2.forward(&h);
        input + &out // Residual connection
    }
}

/// Score Network for estimating ∇_x log p(x)
#[derive(Debug, Clone)]
pub struct ScoreNetwork {
    /// Input projection layer
    input_proj: Layer,
    /// Residual blocks
    res_blocks: Vec<ResidualBlock>,
    /// Output projection layer
    output_proj: Layer,
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
}

impl ScoreNetwork {
    /// Create a new score network
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Dimension of input features
    /// * `hidden_dim` - Hidden layer dimension
    /// * `num_layers` - Number of residual blocks
    pub fn new(input_dim: usize, hidden_dim: usize, num_layers: usize) -> Self {
        let input_proj = Layer::new(input_dim, hidden_dim, ActivationFn::GELU);

        let res_blocks = (0..num_layers)
            .map(|_| ResidualBlock::new(hidden_dim, 4, 0.1))
            .collect();

        let output_proj = Layer::new(hidden_dim, input_dim, ActivationFn::Identity);

        Self {
            input_proj,
            res_blocks,
            output_proj,
            input_dim,
            hidden_dim,
        }
    }

    /// Compute score function s_θ(x)
    ///
    /// # Arguments
    ///
    /// * `x` - Input vector of dimension input_dim
    ///
    /// # Returns
    ///
    /// Score vector of dimension input_dim
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut h = self.input_proj.forward(x);

        for block in &self.res_blocks {
            h = block.forward(&h);
        }

        self.output_proj.forward(&h)
    }

    /// Compute score for batch of inputs
    pub fn forward_batch(&self, x: &Array2<f64>) -> Array2<f64> {
        let batch_size = x.nrows();
        let mut scores = Array2::zeros((batch_size, self.input_dim));

        for i in 0..batch_size {
            let input = x.row(i).to_owned();
            let score = self.forward(&input);
            scores.row_mut(i).assign(&score);
        }

        scores
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;

        // Input projection
        count += self.input_proj.weights.len() + self.input_proj.bias.len();

        // Residual blocks
        for block in &self.res_blocks {
            count += block.layer1.weights.len() + block.layer1.bias.len();
            count += block.layer2.weights.len() + block.layer2.bias.len();
        }

        // Output projection
        count += self.output_proj.weights.len() + self.output_proj.bias.len();

        count
    }
}

/// Multi-scale Score Network with noise level conditioning
#[derive(Debug, Clone)]
pub struct MultiScaleScoreNetwork {
    /// Base score network
    base_network: ScoreNetwork,
    /// Noise level embeddings
    noise_embeddings: Array2<f64>,
    /// Noise levels (sigmas)
    pub sigmas: Array1<f64>,
    /// Number of noise levels
    pub num_noise_levels: usize,
    /// Input dimension
    pub input_dim: usize,
}

impl MultiScaleScoreNetwork {
    /// Create a new multi-scale score network
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Dimension of input features
    /// * `hidden_dim` - Hidden layer dimension
    /// * `num_layers` - Number of residual blocks
    /// * `num_noise_levels` - Number of noise levels
    /// * `sigma_min` - Minimum noise level
    /// * `sigma_max` - Maximum noise level
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        num_noise_levels: usize,
        sigma_min: f64,
        sigma_max: f64,
    ) -> Self {
        let base_network = ScoreNetwork::new(input_dim, hidden_dim, num_layers);

        // Geometric noise schedule
        let sigmas = Array1::from_iter((0..num_noise_levels).map(|i| {
            let t = i as f64 / (num_noise_levels - 1).max(1) as f64;
            sigma_max * (sigma_min / sigma_max).powf(t)
        }));

        // Initialize noise embeddings randomly
        let mut rng = rand::thread_rng();
        let std = (1.0 / hidden_dim as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        let noise_embeddings =
            Array2::from_shape_fn((num_noise_levels, hidden_dim), |_| normal.sample(&mut rng));

        Self {
            base_network,
            noise_embeddings,
            sigmas,
            num_noise_levels,
            input_dim,
        }
    }

    /// Compute score at given noise level
    ///
    /// # Arguments
    ///
    /// * `x` - Input vector
    /// * `noise_level` - Noise level index (0 = highest noise, num_levels-1 = lowest noise)
    ///
    /// # Returns
    ///
    /// Score vector
    pub fn forward(&self, x: &Array1<f64>, noise_level: usize) -> Array1<f64> {
        let noise_level = noise_level.min(self.num_noise_levels - 1);

        // Get noise embedding
        let noise_emb = self.noise_embeddings.row(noise_level).to_owned();

        // Forward through input projection
        let mut h = self.base_network.input_proj.forward(x);

        // Add noise embedding
        h = &h + &noise_emb;

        // Forward through residual blocks
        for block in &self.base_network.res_blocks {
            h = block.forward(&h);
        }

        // Output projection
        self.base_network.output_proj.forward(&h)
    }

    /// Get sigma for given noise level
    pub fn get_sigma(&self, noise_level: usize) -> f64 {
        self.sigmas[noise_level.min(self.num_noise_levels - 1)]
    }

    /// Compute score for batch at given noise level
    pub fn forward_batch(&self, x: &Array2<f64>, noise_level: usize) -> Array2<f64> {
        let batch_size = x.nrows();
        let mut scores = Array2::zeros((batch_size, self.input_dim));

        for i in 0..batch_size {
            let input = x.row(i).to_owned();
            let score = self.forward(&input, noise_level);
            scores.row_mut(i).assign(&score);
        }

        scores
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.base_network.num_parameters() + self.noise_embeddings.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_activation_gelu() {
        let gelu = ActivationFn::GELU;

        // GELU(0) ≈ 0
        assert_relative_eq!(gelu.apply(0.0), 0.0, epsilon = 1e-6);

        // GELU(x) ≈ x for large positive x
        assert_relative_eq!(gelu.apply(3.0), 2.9959, epsilon = 0.01);

        // GELU(x) ≈ 0 for large negative x
        assert_relative_eq!(gelu.apply(-3.0), -0.0040, epsilon = 0.01);
    }

    #[test]
    fn test_layer_forward() {
        let layer = Layer::new(4, 8, ActivationFn::ReLU);
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let output = layer.forward(&input);

        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_score_network_forward() {
        let network = ScoreNetwork::new(10, 32, 2);
        let input = Array1::zeros(10);

        let score = network.forward(&input);

        assert_eq!(score.len(), 10);
    }

    #[test]
    fn test_multiscale_network() {
        let network = MultiScaleScoreNetwork::new(10, 32, 2, 5, 0.01, 1.0);

        assert_eq!(network.num_noise_levels, 5);
        assert_eq!(network.sigmas.len(), 5);

        // Check geometric schedule
        assert_relative_eq!(network.get_sigma(0), 1.0, epsilon = 1e-6);
        assert!(network.get_sigma(4) < network.get_sigma(0));

        let input = Array1::zeros(10);
        let score = network.forward(&input, 2);
        assert_eq!(score.len(), 10);
    }
}
