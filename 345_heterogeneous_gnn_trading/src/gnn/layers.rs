//! Neural network layers for Heterogeneous GNN

use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Activation functions
#[derive(Debug, Clone, Copy)]
pub enum ActivationFn {
    ReLU,
    LeakyReLU(f64),
    Tanh,
    Sigmoid,
    GELU,
    None,
}

impl ActivationFn {
    /// Apply activation function
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            ActivationFn::ReLU => x.max(0.0),
            ActivationFn::LeakyReLU(alpha) => if x > 0.0 { x } else { alpha * x },
            ActivationFn::Tanh => x.tanh(),
            ActivationFn::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFn::GELU => {
                // Approximate GELU
                0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh())
            }
            ActivationFn::None => x,
        }
    }

    /// Apply activation to array
    pub fn apply_array(&self, arr: &Array1<f64>) -> Array1<f64> {
        arr.mapv(|x| self.apply(x))
    }
}

/// Linear layer
#[derive(Debug, Clone)]
pub struct LinearLayer {
    /// Weight matrix
    weights: Array2<f64>,
    /// Bias vector
    bias: Array1<f64>,
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
}

impl LinearLayer {
    /// Create a new linear layer with Xavier initialization
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let weights = Array2::from_shape_fn((output_dim, input_dim), |_| normal.sample(&mut rng));
        let bias = Array1::zeros(output_dim);

        Self {
            weights,
            bias,
            input_dim,
            output_dim,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        self.weights.dot(input) + &self.bias
    }

    /// Forward pass for batch
    pub fn forward_batch(&self, input: &Array2<f64>) -> Array2<f64> {
        input.dot(&self.weights.t()) + &self.bias
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        self.input_dim * self.output_dim + self.output_dim
    }
}

/// HGNN Layer with message passing
#[derive(Debug)]
pub struct HGNNLayer {
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Self transformation
    self_linear: LinearLayer,
    /// Neighbor transformation
    neighbor_linear: LinearLayer,
    /// Output projection
    output_linear: LinearLayer,
    /// Layer normalization parameters
    ln_gamma: Array1<f64>,
    ln_beta: Array1<f64>,
    /// Activation function
    activation: ActivationFn,
}

impl HGNNLayer {
    /// Create a new HGNN layer
    pub fn new(input_dim: usize, output_dim: usize, num_heads: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            num_heads,
            self_linear: LinearLayer::new(input_dim, output_dim),
            neighbor_linear: LinearLayer::new(input_dim, output_dim),
            output_linear: LinearLayer::new(output_dim * 2, output_dim),
            ln_gamma: Array1::ones(output_dim),
            ln_beta: Array1::zeros(output_dim),
            activation: ActivationFn::GELU,
        }
    }

    /// Forward pass for single node without neighbors
    pub fn forward_single(&self, input: &Array1<f64>) -> Array1<f64> {
        let self_transformed = self.self_linear.forward(input);
        let activated = self.activation.apply_array(&self_transformed);
        self.layer_norm(&activated)
    }

    /// Forward pass with neighbor message
    pub fn forward_with_message(&self, input: &Array1<f64>, neighbor_agg: &Array1<f64>) -> Array1<f64> {
        let self_transformed = self.self_linear.forward(input);
        let neighbor_transformed = self.neighbor_linear.forward(neighbor_agg);

        // Concatenate and project
        let mut combined = Array1::zeros(self.output_dim * 2);
        for (i, &val) in self_transformed.iter().enumerate() {
            combined[i] = val;
        }
        for (i, &val) in neighbor_transformed.iter().enumerate() {
            combined[self.output_dim + i] = val;
        }

        let output = self.output_linear.forward(&combined);
        let activated = self.activation.apply_array(&output);

        // Residual connection if dimensions match
        let result = if self.input_dim == self.output_dim {
            &activated + input
        } else {
            activated
        };

        self.layer_norm(&result)
    }

    /// Layer normalization
    fn layer_norm(&self, input: &Array1<f64>) -> Array1<f64> {
        let mean = input.mean().unwrap_or(0.0);
        let variance = input.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(1.0);
        let std = (variance + 1e-5).sqrt();

        input.mapv(|x| (x - mean) / std) * &self.ln_gamma + &self.ln_beta
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        self.self_linear.param_count()
            + self.neighbor_linear.param_count()
            + self.output_linear.param_count()
            + 2 * self.output_dim  // Layer norm params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_functions() {
        assert_eq!(ActivationFn::ReLU.apply(-1.0), 0.0);
        assert_eq!(ActivationFn::ReLU.apply(1.0), 1.0);
        assert!(ActivationFn::Sigmoid.apply(0.0) - 0.5 < 1e-6);
    }

    #[test]
    fn test_linear_layer() {
        let layer = LinearLayer::new(10, 5);
        let input = Array1::ones(10);
        let output = layer.forward(&input);
        assert_eq!(output.len(), 5);
    }

    #[test]
    fn test_hgnn_layer() {
        let layer = HGNNLayer::new(16, 8, 2);
        let input = Array1::ones(16);
        let output = layer.forward_single(&input);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_hgnn_layer_with_message() {
        let layer = HGNNLayer::new(16, 16, 2);
        let input = Array1::ones(16);
        let message = Array1::ones(16) * 0.5;
        let output = layer.forward_with_message(&input, &message);
        assert_eq!(output.len(), 16);
    }
}
