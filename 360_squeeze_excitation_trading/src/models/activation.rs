//! Activation functions for neural networks
//!
//! This module provides common activation functions used in SE networks
//! and other neural network architectures.

use ndarray::Array1;

/// ReLU activation function (element-wise)
///
/// f(x) = max(0, x)
///
/// # Arguments
///
/// * `x` - Input array
///
/// # Returns
///
/// Array with ReLU applied element-wise
#[inline]
pub fn relu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.max(0.0))
}

/// Sigmoid activation function (element-wise)
///
/// f(x) = 1 / (1 + exp(-x))
///
/// # Arguments
///
/// * `x` - Input array
///
/// # Returns
///
/// Array with sigmoid applied element-wise (values in [0, 1])
#[inline]
pub fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

/// Tanh activation function (element-wise)
///
/// f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
///
/// # Arguments
///
/// * `x` - Input array
///
/// # Returns
///
/// Array with tanh applied element-wise (values in [-1, 1])
#[inline]
pub fn tanh(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.tanh())
}

/// Leaky ReLU activation function (element-wise)
///
/// f(x) = x if x > 0, else alpha * x
///
/// # Arguments
///
/// * `x` - Input array
/// * `alpha` - Slope for negative values (typically 0.01)
///
/// # Returns
///
/// Array with Leaky ReLU applied element-wise
#[inline]
pub fn leaky_relu(x: &Array1<f64>, alpha: f64) -> Array1<f64> {
    x.mapv(|v| if v > 0.0 { v } else { alpha * v })
}

/// ELU (Exponential Linear Unit) activation function
///
/// f(x) = x if x > 0, else alpha * (exp(x) - 1)
///
/// # Arguments
///
/// * `x` - Input array
/// * `alpha` - Scale for negative values (typically 1.0)
///
/// # Returns
///
/// Array with ELU applied element-wise
#[inline]
pub fn elu(x: &Array1<f64>, alpha: f64) -> Array1<f64> {
    x.mapv(|v| if v > 0.0 { v } else { alpha * (v.exp() - 1.0) })
}

/// Softmax activation function
///
/// Converts raw scores to probability distribution
///
/// # Arguments
///
/// * `x` - Input array of raw scores
///
/// # Returns
///
/// Array of probabilities that sum to 1
#[inline]
pub fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_x = x.mapv(|v| (v - max_val).exp());
    let sum = exp_x.sum();
    exp_x / sum
}

/// GELU (Gaussian Error Linear Unit) activation function
///
/// Approximation: x * sigmoid(1.702 * x)
///
/// # Arguments
///
/// * `x` - Input array
///
/// # Returns
///
/// Array with GELU applied element-wise
#[inline]
pub fn gelu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v * (1.0 / (1.0 + (-1.702 * v).exp())))
}

/// Swish activation function
///
/// f(x) = x * sigmoid(beta * x)
///
/// # Arguments
///
/// * `x` - Input array
/// * `beta` - Scaling parameter (beta=1 gives standard swish)
///
/// # Returns
///
/// Array with Swish applied element-wise
#[inline]
pub fn swish(x: &Array1<f64>, beta: f64) -> Array1<f64> {
    x.mapv(|v| v / (1.0 + (-beta * v).exp()))
}

/// Hard sigmoid activation (faster approximation)
///
/// f(x) = clip((x + 3) / 6, 0, 1)
///
/// # Arguments
///
/// * `x` - Input array
///
/// # Returns
///
/// Array with hard sigmoid applied element-wise
#[inline]
pub fn hard_sigmoid(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| ((v + 3.0) / 6.0).clamp(0.0, 1.0))
}

/// Scalar activation functions for single values

pub mod scalar {
    /// Scalar ReLU
    #[inline]
    pub fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    /// Scalar sigmoid
    #[inline]
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Scalar tanh
    #[inline]
    pub fn tanh(x: f64) -> f64 {
        x.tanh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_relu() {
        let x = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = relu(&x);
        let expected = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 2.0]);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b);
        }
    }

    #[test]
    fn test_sigmoid() {
        let x = Array1::from_vec(vec![-100.0, 0.0, 100.0]);
        let result = sigmoid(&x);

        assert_relative_eq!(result[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 0.5, epsilon = 1e-10);
        assert_relative_eq!(result[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let result = softmax(&x);

        assert_relative_eq!(result.sum(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tanh_range() {
        let x = Array1::from_vec(vec![-100.0, 0.0, 100.0]);
        let result = tanh(&x);

        assert!(result.iter().all(|&v| v >= -1.0 && v <= 1.0));
    }
}
