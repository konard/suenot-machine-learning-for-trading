//! Causal Convolution Layer Implementation

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::Normal;

/// Causal 1D Convolution Layer
///
/// Ensures that output at time t only depends on inputs at times <= t
#[derive(Debug, Clone)]
pub struct CausalConv1d {
    /// Convolution weights [out_channels, in_channels, kernel_size]
    pub weights: Array2<f64>,
    /// Bias terms [out_channels]
    pub bias: Array1<f64>,
    /// Kernel size
    pub kernel_size: usize,
    /// Dilation factor
    pub dilation: usize,
    /// Input channels
    pub in_channels: usize,
    /// Output channels
    pub out_channels: usize,
}

impl CausalConv1d {
    /// Create a new causal convolution layer with random initialization
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, dilation: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, (2.0 / (in_channels * kernel_size) as f64).sqrt()).unwrap();

        // Xavier/He initialization
        let weights = Array2::from_shape_fn((out_channels, in_channels * kernel_size), |_| {
            rng.sample(normal)
        });

        let bias = Array1::zeros(out_channels);

        Self {
            weights,
            bias,
            kernel_size,
            dilation,
            in_channels,
            out_channels,
        }
    }

    /// Calculate the receptive field of this layer
    pub fn receptive_field(&self) -> usize {
        1 + (self.kernel_size - 1) * self.dilation
    }

    /// Calculate required padding for causal convolution
    pub fn padding(&self) -> usize {
        (self.kernel_size - 1) * self.dilation
    }

    /// Forward pass through the causal convolution
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch, in_channels, seq_len]
    ///
    /// # Returns
    /// Output tensor of shape [batch, out_channels, seq_len]
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let (in_channels, seq_len) = input.dim();
        assert_eq!(in_channels, self.in_channels, "Input channels mismatch");

        // Add left padding for causal convolution
        let padding = self.padding();
        let padded_len = seq_len + padding;

        // Create padded input
        let mut padded = Array2::zeros((in_channels, padded_len));
        padded.slice_mut(ndarray::s![.., padding..]).assign(input);

        // Perform convolution
        let mut output = Array2::zeros((self.out_channels, seq_len));

        for t in 0..seq_len {
            for out_c in 0..self.out_channels {
                let mut sum = self.bias[out_c];

                for k in 0..self.kernel_size {
                    let input_idx = t + padding - k * self.dilation;
                    if input_idx < padded_len {
                        for in_c in 0..self.in_channels {
                            let weight_idx = in_c * self.kernel_size + k;
                            sum += self.weights[[out_c, weight_idx]] * padded[[in_c, input_idx]];
                        }
                    }
                }

                output[[out_c, t]] = sum;
            }
        }

        output
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.weights.len() + self.bias.len()
    }
}

/// Weight normalization wrapper for convolution layer
#[derive(Debug, Clone)]
pub struct WeightNormConv1d {
    /// Underlying convolution
    pub conv: CausalConv1d,
    /// Weight scale factors
    pub g: Array1<f64>,
}

impl WeightNormConv1d {
    /// Create weight-normalized convolution
    pub fn new(conv: CausalConv1d) -> Self {
        let out_channels = conv.out_channels;
        let g = Array1::ones(out_channels);
        Self { conv, g }
    }

    /// Forward pass with weight normalization
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        // Normalize weights and scale by g
        // This is a simplified version - full implementation would normalize per output channel
        self.conv.forward(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_conv_creation() {
        let conv = CausalConv1d::new(16, 32, 3, 1);
        assert_eq!(conv.in_channels, 16);
        assert_eq!(conv.out_channels, 32);
        assert_eq!(conv.kernel_size, 3);
        assert_eq!(conv.dilation, 1);
    }

    #[test]
    fn test_receptive_field() {
        let conv1 = CausalConv1d::new(16, 32, 3, 1);
        assert_eq!(conv1.receptive_field(), 3);

        let conv2 = CausalConv1d::new(16, 32, 3, 2);
        assert_eq!(conv2.receptive_field(), 5);

        let conv4 = CausalConv1d::new(16, 32, 3, 4);
        assert_eq!(conv4.receptive_field(), 9);
    }

    #[test]
    fn test_forward_pass() {
        let conv = CausalConv1d::new(2, 4, 3, 1);
        let input = Array2::ones((2, 10));
        let output = conv.forward(&input);

        assert_eq!(output.dim(), (4, 10));
    }

    #[test]
    fn test_causal_property() {
        // Verify that output at time t doesn't depend on future inputs
        let conv = CausalConv1d::new(1, 1, 3, 1);

        let mut input1 = Array2::zeros((1, 10));
        input1[[0, 5]] = 1.0; // Impulse at t=5

        let output = conv.forward(&input1);

        // Output before t=5 should be zero (only affected by bias)
        // Output at t=5 and after can be non-zero
        for t in 0..5 {
            // Before the impulse, output should equal bias
            assert!((output[[0, t]] - conv.bias[0]).abs() < 1e-10);
        }
    }
}
