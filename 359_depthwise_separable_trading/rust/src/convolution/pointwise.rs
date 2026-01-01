//! Pointwise 1D Convolution (1x1 Convolution)
//!
//! Mixes information across channels at each position.
//! This is the second step of depthwise separable convolution.

use ndarray::{Array1, Array2, Array3, Axis};
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use super::{Activation, ConvError};

/// Pointwise 1D Convolution Layer (1x1 Convolution)
///
/// Applies a 1x1 convolution to mix channel information.
/// Equivalent to a learned linear transformation at each position.
///
/// Total parameters: `in_channels * out_channels + out_channels (bias)`
///
/// # Example
///
/// ```rust
/// use dsc_trading::convolution::PointwiseConv1d;
/// use ndarray::Array2;
///
/// let conv = PointwiseConv1d::new(10, 64).unwrap();
/// let input = Array2::zeros((10, 100)); // 10 channels, 100 timesteps
/// let output = conv.forward(&input);
/// assert_eq!(output.dim(), (64, 100)); // 64 output channels
/// ```
#[derive(Debug, Clone)]
pub struct PointwiseConv1d {
    /// Number of input channels
    pub in_channels: usize,
    /// Number of output channels
    pub out_channels: usize,
    /// Activation function
    pub activation: Activation,
    /// Weight matrix: shape [out_channels, in_channels]
    pub weights: Array2<f64>,
    /// Bias terms: shape [out_channels]
    pub bias: Option<Array1<f64>>,
}

impl PointwiseConv1d {
    /// Create a new pointwise convolution layer
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    ///
    /// # Returns
    /// A new PointwiseConv1d with Xavier initialization
    pub fn new(in_channels: usize, out_channels: usize) -> Result<Self, ConvError> {
        if in_channels == 0 {
            return Err(ConvError::InvalidChannels(in_channels));
        }
        if out_channels == 0 {
            return Err(ConvError::InvalidChannels(out_channels));
        }

        // Xavier initialization
        let std = (2.0 / (in_channels + out_channels) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let weights = Array2::random((out_channels, in_channels), normal);
        let bias = Some(Array1::zeros(out_channels));

        Ok(Self {
            in_channels,
            out_channels,
            activation: Activation::None,
            weights,
            bias,
        })
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

    /// Forward pass
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [in_channels, sequence_length]
    ///
    /// # Returns
    /// Output tensor of shape [out_channels, sequence_length]
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let (channels, seq_len) = input.dim();
        assert_eq!(
            channels, self.in_channels,
            "Input channels mismatch: expected {}, got {}",
            self.in_channels, channels
        );

        // Matrix multiplication: [out_channels, in_channels] @ [in_channels, seq_len]
        let mut output = self.weights.dot(input);

        // Add bias
        if let Some(ref bias) = self.bias {
            for (c, mut row) in output.rows_mut().into_iter().enumerate() {
                row.mapv_inplace(|x| x + bias[c]);
            }
        }

        // Apply activation
        self.activation.apply_array(&mut output);

        output
    }

    /// Forward pass for batch input
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch, in_channels, sequence_length]
    ///
    /// # Returns
    /// Output tensor of shape [batch, out_channels, sequence_length]
    pub fn forward_batch(&self, input: &Array3<f64>) -> Array3<f64> {
        let (batch_size, _channels, seq_len) = input.dim();

        let mut output = Array3::zeros((batch_size, self.out_channels, seq_len));

        for b in 0..batch_size {
            let input_slice = input.index_axis(Axis(0), b).to_owned();
            let output_slice = self.forward(&input_slice);

            for c in 0..self.out_channels {
                for i in 0..seq_len {
                    output[[b, c, i]] = output_slice[[c, i]];
                }
            }
        }

        output
    }

    /// Count number of parameters
    pub fn num_parameters(&self) -> usize {
        let weight_params = self.in_channels * self.out_channels;
        let bias_params = if self.bias.is_some() {
            self.out_channels
        } else {
            0
        };
        weight_params + bias_params
    }

    /// Count number of FLOPs for given sequence length
    pub fn flops(&self, seq_len: usize) -> usize {
        // Matrix multiply-add for each position
        self.in_channels * self.out_channels * seq_len * 2
    }

    /// Create an expansion layer (increases channels)
    pub fn expand(in_channels: usize, expansion_factor: usize) -> Result<Self, ConvError> {
        Self::new(in_channels, in_channels * expansion_factor)
    }

    /// Create a projection layer (decreases channels)
    pub fn project(in_channels: usize, out_channels: usize) -> Result<Self, ConvError> {
        Self::new(in_channels, out_channels)
    }
}

/// Group of pointwise convolutions for multi-head processing
#[derive(Debug, Clone)]
pub struct MultiHeadPointwise {
    /// Individual heads
    pub heads: Vec<PointwiseConv1d>,
    /// Output projection
    pub output_proj: PointwiseConv1d,
}

impl MultiHeadPointwise {
    /// Create multi-head pointwise convolution
    ///
    /// # Arguments
    /// * `in_channels` - Input channels
    /// * `out_channels` - Output channels per head
    /// * `num_heads` - Number of heads
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        num_heads: usize,
    ) -> Result<Self, ConvError> {
        let per_head_out = out_channels / num_heads;

        let heads: Result<Vec<_>, _> = (0..num_heads)
            .map(|_| PointwiseConv1d::new(in_channels, per_head_out))
            .collect();

        let output_proj = PointwiseConv1d::new(out_channels, out_channels)?;

        Ok(Self {
            heads: heads?,
            output_proj,
        })
    }

    /// Forward pass
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let seq_len = input.dim().1;
        let total_out = self.heads.len() * self.heads[0].out_channels;

        let mut concat = Array2::zeros((total_out, seq_len));

        let mut offset = 0;
        for head in &self.heads {
            let head_out = head.forward(input);
            for c in 0..head.out_channels {
                for i in 0..seq_len {
                    concat[[offset + c, i]] = head_out[[c, i]];
                }
            }
            offset += head.out_channels;
        }

        self.output_proj.forward(&concat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pointwise_creation() {
        let conv = PointwiseConv1d::new(10, 64).unwrap();
        assert_eq!(conv.in_channels, 10);
        assert_eq!(conv.out_channels, 64);
        assert_eq!(conv.weights.dim(), (64, 10));
    }

    #[test]
    fn test_pointwise_invalid_channels() {
        assert!(PointwiseConv1d::new(0, 64).is_err());
        assert!(PointwiseConv1d::new(10, 0).is_err());
    }

    #[test]
    fn test_pointwise_forward() {
        let conv = PointwiseConv1d::new(10, 64).unwrap();
        let input = Array2::ones((10, 100));
        let output = conv.forward(&input);

        assert_eq!(output.dim(), (64, 100));
    }

    #[test]
    fn test_pointwise_expansion() {
        let conv = PointwiseConv1d::expand(32, 4).unwrap();
        assert_eq!(conv.out_channels, 128);
    }

    #[test]
    fn test_parameter_count() {
        let conv = PointwiseConv1d::new(64, 128).unwrap();
        // weights: 64 * 128 = 8192, bias: 128, total: 8320
        assert_eq!(conv.num_parameters(), 8320);
    }

    #[test]
    fn test_pointwise_batch() {
        let conv = PointwiseConv1d::new(10, 32).unwrap();
        let input = Array3::ones((8, 10, 100)); // batch=8
        let output = conv.forward_batch(&input);

        assert_eq!(output.dim(), (8, 32, 100));
    }
}
