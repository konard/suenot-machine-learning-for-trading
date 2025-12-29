//! Activation Functions for Neural Networks
//!
//! Implements common activation functions and their derivatives
//! for use in backpropagation.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Types of activation functions available
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ActivationType {
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Hyperbolic tangent: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Tanh,
    /// Softmax: exp(x_i) / sum(exp(x_j))
    Softmax,
    /// Linear (identity): x
    Linear,
    /// Leaky ReLU: max(0.01x, x)
    LeakyReLU,
}

/// Activation function trait with forward and backward passes
pub trait Activation: Send + Sync {
    /// Apply the activation function
    fn forward(&self, x: &Array1<f64>) -> Array1<f64>;

    /// Compute the derivative for backpropagation
    fn backward(&self, x: &Array1<f64>) -> Array1<f64>;

    /// Apply to 2D array (batch)
    fn forward_batch(&self, x: &Array2<f64>) -> Array2<f64>;

    /// Derivative for batch
    fn backward_batch(&self, x: &Array2<f64>) -> Array2<f64>;
}

/// ReLU activation function
pub struct ReLU;

impl Activation for ReLU {
    fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| v.max(0.0))
    }

    fn backward(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }

    fn forward_batch(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| v.max(0.0))
    }

    fn backward_batch(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }
}

/// Sigmoid activation function
pub struct Sigmoid;

impl Activation for Sigmoid {
    fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn backward(&self, x: &Array1<f64>) -> Array1<f64> {
        let s = self.forward(x);
        &s * &(1.0 - &s)
    }

    fn forward_batch(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn backward_batch(&self, x: &Array2<f64>) -> Array2<f64> {
        let s = self.forward_batch(x);
        &s * &(1.0 - &s)
    }
}

/// Tanh activation function
pub struct TanhActivation;

impl Activation for TanhActivation {
    fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| v.tanh())
    }

    fn backward(&self, x: &Array1<f64>) -> Array1<f64> {
        let t = self.forward(x);
        1.0 - &t * &t
    }

    fn forward_batch(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| v.tanh())
    }

    fn backward_batch(&self, x: &Array2<f64>) -> Array2<f64> {
        let t = self.forward_batch(x);
        1.0 - &t * &t
    }
}

/// Linear (identity) activation function
pub struct Linear;

impl Activation for Linear {
    fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        x.clone()
    }

    fn backward(&self, _x: &Array1<f64>) -> Array1<f64> {
        Array1::ones(_x.len())
    }

    fn forward_batch(&self, x: &Array2<f64>) -> Array2<f64> {
        x.clone()
    }

    fn backward_batch(&self, x: &Array2<f64>) -> Array2<f64> {
        Array2::ones(x.dim())
    }
}

/// Leaky ReLU activation function
pub struct LeakyReLU {
    pub alpha: f64,
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self { alpha: 0.01 }
    }
}

impl Activation for LeakyReLU {
    fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| if v > 0.0 { v } else { self.alpha * v })
    }

    fn backward(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { self.alpha })
    }

    fn forward_batch(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| if v > 0.0 { v } else { self.alpha * v })
    }

    fn backward_batch(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { self.alpha })
    }
}

/// Create an activation function from type
pub fn create_activation(activation_type: ActivationType) -> Box<dyn Activation> {
    match activation_type {
        ActivationType::ReLU => Box::new(ReLU),
        ActivationType::Sigmoid => Box::new(Sigmoid),
        ActivationType::Tanh => Box::new(TanhActivation),
        ActivationType::Linear => Box::new(Linear),
        ActivationType::LeakyReLU => Box::new(LeakyReLU::default()),
        ActivationType::Softmax => Box::new(Sigmoid), // Simplified for now
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_relu() {
        let relu = ReLU;
        let x = Array1::from_vec(vec![-1.0, 0.0, 1.0, 2.0]);
        let y = relu.forward(&x);
        assert_eq!(y, Array1::from_vec(vec![0.0, 0.0, 1.0, 2.0]));
    }

    #[test]
    fn test_sigmoid() {
        let sigmoid = Sigmoid;
        let x = Array1::from_vec(vec![0.0]);
        let y = sigmoid.forward(&x);
        assert_relative_eq!(y[0], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_tanh() {
        let tanh = TanhActivation;
        let x = Array1::from_vec(vec![0.0]);
        let y = tanh.forward(&x);
        assert_relative_eq!(y[0], 0.0, epsilon = 1e-10);
    }
}
