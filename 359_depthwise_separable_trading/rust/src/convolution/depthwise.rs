//! Depthwise 1D Convolution
//!
//! Applies a separate filter to each input channel independently.
//! This is the first step of depthwise separable convolution.

use ndarray::{Array1, Array2, Array3, Axis};
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use super::{Activation, ConvError, Padding};

/// Depthwise 1D Convolution Layer
///
/// Applies a separate convolution filter to each input channel.
/// Total parameters: `channels * kernel_size`
///
/// # Example
///
/// ```rust
/// use dsc_trading::convolution::DepthwiseConv1d;
/// use ndarray::Array2;
///
/// let conv = DepthwiseConv1d::new(10, 3).unwrap();
/// let input = Array2::zeros((10, 100)); // 10 channels, 100 timesteps
/// let output = conv.forward(&input);
/// ```
#[derive(Debug, Clone)]
pub struct DepthwiseConv1d {
    /// Number of input channels
    pub in_channels: usize,
    /// Kernel size
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Dilation factor
    pub dilation: usize,
    /// Padding mode
    pub padding: Padding,
    /// Activation function
    pub activation: Activation,
    /// Convolution weights: shape [channels, kernel_size]
    pub weights: Array2<f64>,
    /// Bias terms: shape [channels]
    pub bias: Option<Array1<f64>>,
}

impl DepthwiseConv1d {
    /// Create a new depthwise convolution layer
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `kernel_size` - Size of the convolution kernel
    ///
    /// # Returns
    /// A new DepthwiseConv1d with Xavier initialization
    pub fn new(in_channels: usize, kernel_size: usize) -> Result<Self, ConvError> {
        if kernel_size == 0 || kernel_size % 2 == 0 {
            return Err(ConvError::InvalidKernelSize(kernel_size));
        }
        if in_channels == 0 {
            return Err(ConvError::InvalidChannels(in_channels));
        }

        // Xavier initialization
        let std = (2.0 / (in_channels * kernel_size) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let weights = Array2::random((in_channels, kernel_size), normal);
        let bias = Some(Array1::zeros(in_channels));

        Ok(Self {
            in_channels,
            kernel_size,
            stride: 1,
            dilation: 1,
            padding: Padding::Same,
            activation: Activation::None,
            weights,
            bias,
        })
    }

    /// Builder method to set stride
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Builder method to set dilation
    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    /// Builder method to set padding
    pub fn with_padding(mut self, padding: Padding) -> Self {
        self.padding = padding;
        self
    }

    /// Builder method to set activation
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Builder method to disable bias
    pub fn without_bias(mut self) -> Self {
        self.bias = None;
        self
    }

    /// Calculate output length for given input length
    pub fn output_length(&self, input_length: usize) -> usize {
        let pad = self.padding.calculate(self.kernel_size, self.dilation);
        let effective_kernel = (self.kernel_size - 1) * self.dilation + 1;
        (input_length + 2 * pad - effective_kernel) / self.stride + 1
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [channels, sequence_length]
    ///
    /// # Returns
    /// Output tensor of shape [channels, output_length]
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let (channels, seq_len) = input.dim();
        assert_eq!(
            channels, self.in_channels,
            "Input channels mismatch: expected {}, got {}",
            self.in_channels, channels
        );

        let pad = self.padding.calculate(self.kernel_size, self.dilation);
        let out_len = self.output_length(seq_len);

        // Pad input if necessary
        let padded = if pad > 0 {
            let mut padded = Array2::zeros((channels, seq_len + 2 * pad));
            for c in 0..channels {
                for i in 0..seq_len {
                    padded[[c, i + pad]] = input[[c, i]];
                }
            }
            padded
        } else {
            input.clone()
        };

        // Perform convolution
        let mut output = Array2::zeros((channels, out_len));

        for c in 0..channels {
            for i in 0..out_len {
                let start = i * self.stride;
                let mut sum = 0.0;

                for k in 0..self.kernel_size {
                    let idx = start + k * self.dilation;
                    if idx < padded.dim().1 {
                        sum += padded[[c, idx]] * self.weights[[c, k]];
                    }
                }

                // Add bias
                if let Some(ref bias) = self.bias {
                    sum += bias[c];
                }

                output[[c, i]] = self.activation.apply(sum);
            }
        }

        output
    }

    /// Forward pass for batch input
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch, channels, sequence_length]
    ///
    /// # Returns
    /// Output tensor of shape [batch, channels, output_length]
    pub fn forward_batch(&self, input: &Array3<f64>) -> Array3<f64> {
        let (batch_size, channels, seq_len) = input.dim();
        let out_len = self.output_length(seq_len);

        let mut output = Array3::zeros((batch_size, channels, out_len));

        for b in 0..batch_size {
            let input_slice = input.index_axis(Axis(0), b).to_owned();
            let output_slice = self.forward(&input_slice);

            for c in 0..channels {
                for i in 0..out_len {
                    output[[b, c, i]] = output_slice[[c, i]];
                }
            }
        }

        output
    }

    /// Count number of parameters
    pub fn num_parameters(&self) -> usize {
        let weight_params = self.in_channels * self.kernel_size;
        let bias_params = if self.bias.is_some() {
            self.in_channels
        } else {
            0
        };
        weight_params + bias_params
    }

    /// Count number of FLOPs for given input length
    pub fn flops(&self, input_length: usize) -> usize {
        let out_len = self.output_length(input_length);
        // Multiply-add for each output position
        self.in_channels * out_len * self.kernel_size * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_depthwise_creation() {
        let conv = DepthwiseConv1d::new(10, 3).unwrap();
        assert_eq!(conv.in_channels, 10);
        assert_eq!(conv.kernel_size, 3);
        assert_eq!(conv.weights.dim(), (10, 3));
    }

    #[test]
    fn test_depthwise_invalid_kernel() {
        assert!(DepthwiseConv1d::new(10, 0).is_err());
        assert!(DepthwiseConv1d::new(10, 2).is_err()); // even kernel
    }

    #[test]
    fn test_depthwise_forward() {
        let conv = DepthwiseConv1d::new(4, 3).unwrap();
        let input = Array2::ones((4, 100));
        let output = conv.forward(&input);

        assert_eq!(output.dim(), (4, 100)); // Same padding
    }

    #[test]
    fn test_depthwise_valid_padding() {
        let conv = DepthwiseConv1d::new(4, 3)
            .unwrap()
            .with_padding(Padding::Valid);
        let input = Array2::ones((4, 100));
        let output = conv.forward(&input);

        assert_eq!(output.dim(), (4, 98)); // 100 - 3 + 1
    }

    #[test]
    fn test_depthwise_with_stride() {
        let conv = DepthwiseConv1d::new(4, 3)
            .unwrap()
            .with_stride(2);
        let input = Array2::ones((4, 100));
        let output = conv.forward(&input);

        assert_eq!(output.dim(), (4, 50)); // 100 / 2
    }

    #[test]
    fn test_parameter_count() {
        let conv = DepthwiseConv1d::new(64, 3).unwrap();
        // weights: 64 * 3 = 192, bias: 64, total: 256
        assert_eq!(conv.num_parameters(), 256);
    }
}
