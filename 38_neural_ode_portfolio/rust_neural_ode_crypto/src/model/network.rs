//! # Neural Network Primitives
//!
//! Basic building blocks for neural networks in pure Rust.

use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Activation functions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Hyperbolic tangent: tanh(x)
    Tanh,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Swish/SiLU: x * sigmoid(x)
    SiLU,
    /// No activation
    Identity,
    /// Softmax (for output layer)
    Softmax,
}

impl Activation {
    /// Apply activation function element-wise
    pub fn apply(&self, x: &Array1<f64>) -> Array1<f64> {
        match self {
            Activation::ReLU => x.mapv(|v| v.max(0.0)),
            Activation::Tanh => x.mapv(|v| v.tanh()),
            Activation::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::SiLU => x.mapv(|v| v / (1.0 + (-v).exp())),
            Activation::Identity => x.clone(),
            Activation::Softmax => {
                let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_x: Array1<f64> = x.mapv(|v| (v - max_val).exp());
                let sum: f64 = exp_x.sum();
                exp_x / sum
            }
        }
    }

    /// Compute derivative for backpropagation
    pub fn derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        match self {
            Activation::ReLU => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            Activation::Tanh => x.mapv(|v| 1.0 - v.tanh().powi(2)),
            Activation::Sigmoid => {
                let s = self.apply(x);
                &s * &(1.0 - &s)
            }
            Activation::SiLU => {
                let sig = x.mapv(|v| 1.0 / (1.0 + (-v).exp()));
                &sig + x * &sig * &(1.0 - &sig)
            }
            Activation::Identity => Array1::ones(x.len()),
            Activation::Softmax => {
                // Jacobian is complex; simplified for now
                let s = self.apply(x);
                &s * &(1.0 - &s)
            }
        }
    }
}

/// Single dense (fully connected) layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    /// Weight matrix (output_dim x input_dim)
    pub weights: Array2<f64>,
    /// Bias vector (output_dim)
    pub bias: Array1<f64>,
    /// Activation function
    pub activation: Activation,
}

impl Layer {
    /// Create a new layer with random initialization
    pub fn new(input_dim: usize, output_dim: usize, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier/Glorot initialization
        let std_dev = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();

        let weights = Array2::from_shape_fn((output_dim, input_dim), |_| {
            normal.sample(&mut rng)
        });

        let bias = Array1::zeros(output_dim);

        Self {
            weights,
            bias,
            activation,
        }
    }

    /// Create layer with specific weights
    pub fn with_weights(
        weights: Array2<f64>,
        bias: Array1<f64>,
        activation: Activation,
    ) -> Self {
        Self {
            weights,
            bias,
            activation,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let z = self.weights.dot(x) + &self.bias;
        self.activation.apply(&z)
    }

    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        self.weights.ncols()
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.weights.nrows()
    }

    /// Get total number of parameters
    pub fn num_params(&self) -> usize {
        self.weights.len() + self.bias.len()
    }
}

/// Multi-Layer Perceptron (MLP)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLP {
    /// List of layers
    pub layers: Vec<Layer>,
}

impl MLP {
    /// Create a new MLP with specified architecture
    ///
    /// # Arguments
    ///
    /// * `dims` - Vector of layer dimensions [input, hidden1, hidden2, ..., output]
    /// * `hidden_activation` - Activation for hidden layers
    /// * `output_activation` - Activation for output layer
    pub fn new(
        dims: &[usize],
        hidden_activation: Activation,
        output_activation: Activation,
    ) -> Self {
        assert!(dims.len() >= 2, "Need at least input and output dimensions");

        let mut layers = Vec::with_capacity(dims.len() - 1);

        for i in 0..dims.len() - 1 {
            let activation = if i == dims.len() - 2 {
                output_activation
            } else {
                hidden_activation
            };

            layers.push(Layer::new(dims[i], dims[i + 1], activation));
        }

        Self { layers }
    }

    /// Create MLP for portfolio dynamics
    pub fn for_portfolio(n_assets: usize, hidden_dim: usize) -> Self {
        // Input: state (hidden_dim) + time (1)
        // Output: drift (hidden_dim)
        let dims = vec![hidden_dim + 1, hidden_dim, hidden_dim, hidden_dim];
        Self::new(&dims, Activation::Tanh, Activation::Identity)
    }

    /// Forward pass through all layers
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut output = x.clone();
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        output
    }

    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        self.layers.first().map(|l| l.input_dim()).unwrap_or(0)
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.layers.last().map(|l| l.output_dim()).unwrap_or(0)
    }

    /// Get total number of parameters
    pub fn num_params(&self) -> usize {
        self.layers.iter().map(|l| l.num_params()).sum()
    }

    /// Get all parameters as a flat vector
    pub fn get_params(&self) -> Vec<f64> {
        let mut params = Vec::with_capacity(self.num_params());
        for layer in &self.layers {
            params.extend(layer.weights.iter());
            params.extend(layer.bias.iter());
        }
        params
    }

    /// Set parameters from a flat vector
    pub fn set_params(&mut self, params: &[f64]) {
        let mut idx = 0;
        for layer in &mut self.layers {
            let w_len = layer.weights.len();
            let b_len = layer.bias.len();

            for (i, w) in layer.weights.iter_mut().enumerate() {
                *w = params[idx + i];
            }
            idx += w_len;

            for (i, b) in layer.bias.iter_mut().enumerate() {
                *b = params[idx + i];
            }
            idx += b_len;
        }
    }

    /// Apply random perturbation to parameters (for simple training)
    pub fn perturb(&mut self, scale: f64) {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, scale).unwrap();

        for layer in &mut self.layers {
            for w in layer.weights.iter_mut() {
                *w += normal.sample(&mut rng);
            }
            for b in layer.bias.iter_mut() {
                *b += normal.sample(&mut rng);
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activations() {
        let x = Array1::from_vec(vec![-1.0, 0.0, 1.0, 2.0]);

        // ReLU
        let relu = Activation::ReLU.apply(&x);
        assert_eq!(relu[0], 0.0);
        assert_eq!(relu[2], 1.0);

        // Tanh
        let tanh = Activation::Tanh.apply(&x);
        assert!(tanh[0] < 0.0);
        assert!((tanh[1]).abs() < 1e-10);

        // Sigmoid
        let sig = Activation::Sigmoid.apply(&x);
        assert!(sig[0] < 0.5);
        assert!((sig[1] - 0.5).abs() < 1e-10);
        assert!(sig[2] > 0.5);

        // Softmax
        let softmax = Activation::Softmax.apply(&x);
        assert!((softmax.sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_layer() {
        let layer = Layer::new(3, 2, Activation::ReLU);
        assert_eq!(layer.input_dim(), 3);
        assert_eq!(layer.output_dim(), 2);

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = layer.forward(&x);
        assert_eq!(y.len(), 2);
    }

    #[test]
    fn test_mlp() {
        let mlp = MLP::new(&[4, 8, 8, 2], Activation::Tanh, Activation::Identity);

        assert_eq!(mlp.input_dim(), 4);
        assert_eq!(mlp.output_dim(), 2);
        assert_eq!(mlp.layers.len(), 3);

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let y = mlp.forward(&x);
        assert_eq!(y.len(), 2);
    }

    #[test]
    fn test_mlp_params() {
        let mut mlp = MLP::new(&[2, 3, 2], Activation::ReLU, Activation::Identity);
        let n_params = mlp.num_params();

        // Layer 1: 2*3 weights + 3 bias = 9
        // Layer 2: 3*2 weights + 2 bias = 8
        // Total: 17
        assert_eq!(n_params, 17);

        let params = mlp.get_params();
        assert_eq!(params.len(), 17);

        // Test setting params
        let new_params: Vec<f64> = (0..17).map(|i| i as f64).collect();
        mlp.set_params(&new_params);

        let retrieved = mlp.get_params();
        for (a, b) in new_params.iter().zip(retrieved.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_softmax_stability() {
        // Test with large values (should not overflow)
        let x = Array1::from_vec(vec![1000.0, 1001.0, 1002.0]);
        let softmax = Activation::Softmax.apply(&x);

        assert!(!softmax[0].is_nan());
        assert!(!softmax[0].is_infinite());
        assert!((softmax.sum() - 1.0).abs() < 1e-10);
    }
}
