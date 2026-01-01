//! Convolution operations module
//!
//! This module provides efficient 1D convolution implementations:
//! - Depthwise convolution: applies a separate filter to each input channel
//! - Pointwise convolution: 1x1 convolution to mix channel information
//! - Depthwise Separable: combines both for efficient computation

mod depthwise;
mod pointwise;
mod separable;

pub use depthwise::DepthwiseConv1d;
pub use pointwise::PointwiseConv1d;
pub use separable::DepthwiseSeparableConv1d;

use ndarray::{Array1, Array2, Array3};
use thiserror::Error;

/// Activation functions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    /// No activation (identity)
    None,
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Leaky ReLU: max(alpha * x, x)
    LeakyReLU(f64),
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Swish: x * sigmoid(x)
    Swish,
}

impl Activation {
    /// Apply activation function element-wise
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::None => x,
            Activation::ReLU => x.max(0.0),
            Activation::LeakyReLU(alpha) => {
                if x > 0.0 { x } else { alpha * x }
            }
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Swish => x / (1.0 + (-x).exp()),
        }
    }

    /// Apply activation to an array
    pub fn apply_array(&self, arr: &mut Array2<f64>) {
        arr.mapv_inplace(|x| self.apply(x));
    }
}

/// Convolution errors
#[derive(Error, Debug)]
pub enum ConvError {
    #[error("Invalid kernel size: {0}. Must be positive and odd.")]
    InvalidKernelSize(usize),

    #[error("Invalid channel count: {0}. Must be positive.")]
    InvalidChannels(usize),

    #[error("Invalid dilation: {0}. Must be positive.")]
    InvalidDilation(usize),

    #[error("Input shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("Computation error: {0}")]
    ComputationError(String),
}

/// Padding mode for convolutions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Padding {
    /// No padding
    Valid,
    /// Pad to maintain input size
    Same,
    /// Custom padding amount
    Custom(usize),
}

impl Padding {
    /// Calculate padding size for given kernel and dilation
    pub fn calculate(&self, kernel_size: usize, dilation: usize) -> usize {
        match self {
            Padding::Valid => 0,
            Padding::Same => (kernel_size - 1) * dilation / 2,
            Padding::Custom(p) => *p,
        }
    }
}

/// Layer normalization for convolution outputs
pub fn layer_norm(input: &Array2<f64>, eps: f64) -> Array2<f64> {
    let mut output = input.clone();

    for mut row in output.rows_mut() {
        let mean = row.mean().unwrap_or(0.0);
        let var = row.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(1.0);
        let std = (var + eps).sqrt();

        row.mapv_inplace(|x| (x - mean) / std);
    }

    output
}

/// Batch normalization parameters
#[derive(Debug, Clone)]
pub struct BatchNorm1d {
    /// Running mean per channel
    pub running_mean: Array1<f64>,
    /// Running variance per channel
    pub running_var: Array1<f64>,
    /// Scale parameter (gamma)
    pub gamma: Array1<f64>,
    /// Shift parameter (beta)
    pub beta: Array1<f64>,
    /// Small constant for numerical stability
    pub eps: f64,
    /// Momentum for running stats update
    pub momentum: f64,
}

impl BatchNorm1d {
    /// Create new batch normalization layer
    pub fn new(num_channels: usize) -> Self {
        Self {
            running_mean: Array1::zeros(num_channels),
            running_var: Array1::ones(num_channels),
            gamma: Array1::ones(num_channels),
            beta: Array1::zeros(num_channels),
            eps: 1e-5,
            momentum: 0.1,
        }
    }

    /// Apply batch normalization (inference mode)
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let mut output = Array2::zeros(input.raw_dim());

        for (c, mut col) in output.columns_mut().into_iter().enumerate() {
            let mean = self.running_mean[c];
            let var = self.running_var[c];
            let gamma = self.gamma[c];
            let beta = self.beta[c];

            for (i, val) in input.column(c).iter().enumerate() {
                col[i] = gamma * (val - mean) / (var + self.eps).sqrt() + beta;
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_activation_relu() {
        let act = Activation::ReLU;
        assert_eq!(act.apply(5.0), 5.0);
        assert_eq!(act.apply(-3.0), 0.0);
        assert_eq!(act.apply(0.0), 0.0);
    }

    #[test]
    fn test_activation_sigmoid() {
        let act = Activation::Sigmoid;
        assert_relative_eq!(act.apply(0.0), 0.5, epsilon = 1e-10);
        assert!(act.apply(100.0) > 0.99);
        assert!(act.apply(-100.0) < 0.01);
    }

    #[test]
    fn test_padding_calculation() {
        assert_eq!(Padding::Valid.calculate(3, 1), 0);
        assert_eq!(Padding::Same.calculate(3, 1), 1);
        assert_eq!(Padding::Same.calculate(5, 1), 2);
        assert_eq!(Padding::Same.calculate(3, 2), 2);
        assert_eq!(Padding::Custom(5).calculate(3, 1), 5);
    }
}
