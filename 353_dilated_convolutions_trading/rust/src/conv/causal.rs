//! Causal 1D Convolution Implementation
//!
//! Causal convolutions ensure that the output at time t only depends
//! on inputs at times <= t. This is essential for online prediction.

use ndarray::{Array1, Array2};
use rand::Rng;

/// Causal 1D Convolution Layer
///
/// Standard convolution with left padding to ensure causality.
/// This is a special case of DilatedConv1D with dilation=1.
#[derive(Debug, Clone)]
pub struct CausalConv1D {
    /// Input channels
    in_channels: usize,
    /// Output channels
    out_channels: usize,
    /// Kernel size
    kernel_size: usize,
    /// Weights: shape (out_channels, in_channels, kernel_size)
    weights: Vec<Array2<f64>>,
    /// Bias: shape (out_channels,)
    bias: Array1<f64>,
}

impl CausalConv1D {
    /// Create a new causal convolution layer
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (in_channels * kernel_size) as f64).sqrt();

        let weights = (0..out_channels)
            .map(|_| {
                Array2::from_shape_fn((in_channels, kernel_size), |_| {
                    rng.gen::<f64>() * std * 2.0 - std
                })
            })
            .collect();

        let bias = Array1::zeros(out_channels);

        Self {
            in_channels,
            out_channels,
            kernel_size,
            weights,
            bias,
        }
    }

    /// Create with specific weights
    pub fn with_weights(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        weights: Vec<Array2<f64>>,
        bias: Array1<f64>,
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            weights,
            bias,
        }
    }

    /// Get the receptive field (same as kernel size for causal conv)
    pub fn receptive_field(&self) -> usize {
        self.kernel_size
    }

    /// Apply the causal convolution
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let (n_channels, seq_len) = input.dim();
        assert_eq!(n_channels, self.in_channels);

        // Padding for causality
        let padding = self.kernel_size - 1;
        let padded_len = seq_len + padding;

        let mut padded = Array2::zeros((n_channels, padded_len));
        padded
            .slice_mut(ndarray::s![.., padding..])
            .assign(input);

        let mut output = Array2::zeros((self.out_channels, seq_len));

        for (out_idx, weight) in self.weights.iter().enumerate() {
            for t in 0..seq_len {
                let mut sum = 0.0;

                for k in 0..self.kernel_size {
                    let input_idx = t + padding - k;
                    for c in 0..self.in_channels {
                        sum += weight[[c, k]] * padded[[c, input_idx]];
                    }
                }

                output[[out_idx, t]] = sum + self.bias[out_idx];
            }
        }

        output
    }
}

/// 1x1 Convolution (Pointwise convolution)
///
/// Used for changing the number of channels without changing
/// the sequence length. Equivalent to a linear transformation
/// applied independently to each timestep.
#[derive(Debug, Clone)]
pub struct Conv1x1 {
    /// Input channels
    in_channels: usize,
    /// Output channels
    out_channels: usize,
    /// Weights: shape (out_channels, in_channels)
    weights: Array2<f64>,
    /// Bias: shape (out_channels,)
    bias: Array1<f64>,
}

impl Conv1x1 {
    /// Create a new 1x1 convolution layer
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / in_channels as f64).sqrt();

        let weights = Array2::from_shape_fn((out_channels, in_channels), |_| {
            rng.gen::<f64>() * std * 2.0 - std
        });

        let bias = Array1::zeros(out_channels);

        Self {
            in_channels,
            out_channels,
            weights,
            bias,
        }
    }

    /// Create with specific weights
    pub fn with_weights(
        in_channels: usize,
        out_channels: usize,
        weights: Array2<f64>,
        bias: Array1<f64>,
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            weights,
            bias,
        }
    }

    /// Apply the 1x1 convolution
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let (n_channels, seq_len) = input.dim();
        assert_eq!(n_channels, self.in_channels);

        // Linear transformation: output = weights @ input + bias
        let mut output = self.weights.dot(input);

        // Add bias
        for t in 0..seq_len {
            for c in 0..self.out_channels {
                output[[c, t]] += self.bias[c];
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_conv_shape() {
        let conv = CausalConv1D::new(5, 10, 3);
        let input = Array2::zeros((5, 50));
        let output = conv.forward(&input);
        assert_eq!(output.dim(), (10, 50));
    }

    #[test]
    fn test_conv1x1_shape() {
        let conv = Conv1x1::new(32, 16);
        let input = Array2::zeros((32, 100));
        let output = conv.forward(&input);
        assert_eq!(output.dim(), (16, 100));
    }

    #[test]
    fn test_conv1x1_is_pointwise() {
        // Verify that each timestep is processed independently
        let conv = Conv1x1::new(2, 2);

        let mut input1 = Array2::zeros((2, 3));
        input1[[0, 1]] = 1.0;

        let mut input2 = Array2::zeros((2, 3));
        input2[[0, 1]] = 1.0;
        input2[[0, 0]] = 999.0; // Change a different timestep

        let output1 = conv.forward(&input1);
        let output2 = conv.forward(&input2);

        // Output at t=1 should be the same
        assert!((output1[[0, 1]] - output2[[0, 1]]).abs() < 1e-10);
        assert!((output1[[1, 1]] - output2[[1, 1]]).abs() < 1e-10);
    }
}
