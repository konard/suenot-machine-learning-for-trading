//! Dilated 1D Convolution Implementation
//!
//! Dilated convolutions insert gaps between kernel elements,
//! allowing the network to have an exponentially large receptive field.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;

/// Dilated 1D Convolution Layer
///
/// Applies a dilated convolution operation:
/// y[t] = Σᵢ w[i] · x[t - i·d]
///
/// where d is the dilation rate.
#[derive(Debug, Clone)]
pub struct DilatedConv1D {
    /// Input channels
    in_channels: usize,
    /// Output channels
    out_channels: usize,
    /// Kernel size
    kernel_size: usize,
    /// Dilation rate
    dilation: usize,
    /// Weights: shape (out_channels, in_channels, kernel_size)
    weights: Vec<Array2<f64>>,
    /// Bias: shape (out_channels,)
    bias: Array1<f64>,
}

impl DilatedConv1D {
    /// Create a new dilated convolution layer with random initialization
    ///
    /// # Arguments
    /// - `in_channels` - Number of input channels
    /// - `out_channels` - Number of output channels
    /// - `kernel_size` - Size of the convolution kernel
    /// - `dilation` - Dilation rate (1 = standard convolution)
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization
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
            dilation,
            weights,
            bias,
        }
    }

    /// Create with specific weights (for testing or loading pretrained)
    pub fn with_weights(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        weights: Vec<Array2<f64>>,
        bias: Array1<f64>,
    ) -> Self {
        assert_eq!(weights.len(), out_channels);
        assert_eq!(bias.len(), out_channels);
        for w in &weights {
            assert_eq!(w.shape(), &[in_channels, kernel_size]);
        }

        Self {
            in_channels,
            out_channels,
            kernel_size,
            dilation,
            weights,
            bias,
        }
    }

    /// Get the effective receptive field size
    pub fn receptive_field(&self) -> usize {
        (self.kernel_size - 1) * self.dilation + 1
    }

    /// Get the dilation rate
    pub fn dilation(&self) -> usize {
        self.dilation
    }

    /// Get the kernel size
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Apply the dilated convolution (causal, left-padded)
    ///
    /// # Arguments
    /// - `input` - Input tensor of shape (in_channels, sequence_length)
    ///
    /// # Returns
    /// - Output tensor of shape (out_channels, sequence_length)
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let (n_channels, seq_len) = input.dim();
        assert_eq!(n_channels, self.in_channels, "Input channel mismatch");

        // Calculate padding for causal convolution
        let padding = (self.kernel_size - 1) * self.dilation;

        // Create padded input
        let padded_len = seq_len + padding;
        let mut padded = Array2::zeros((n_channels, padded_len));
        padded
            .slice_mut(ndarray::s![.., padding..])
            .assign(input);

        // Output tensor
        let mut output = Array2::zeros((self.out_channels, seq_len));

        // Apply convolution for each output channel
        for (out_idx, weight) in self.weights.iter().enumerate() {
            for t in 0..seq_len {
                let mut sum = 0.0;

                // Convolve over kernel
                for k in 0..self.kernel_size {
                    let input_idx = t + padding - k * self.dilation;

                    // Sum over input channels
                    for c in 0..self.in_channels {
                        sum += weight[[c, k]] * padded[[c, input_idx]];
                    }
                }

                output[[out_idx, t]] = sum + self.bias[out_idx];
            }
        }

        output
    }

    /// Apply ReLU activation
    pub fn relu(input: &Array2<f64>) -> Array2<f64> {
        input.mapv(|x| x.max(0.0))
    }

    /// Apply Tanh activation
    pub fn tanh(input: &Array2<f64>) -> Array2<f64> {
        input.mapv(|x| x.tanh())
    }

    /// Apply Sigmoid activation
    pub fn sigmoid(input: &Array2<f64>) -> Array2<f64> {
        input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }
}

/// Multi-scale dilated convolution block
///
/// Applies multiple dilated convolutions with different dilation rates
/// and concatenates the outputs.
#[derive(Debug, Clone)]
pub struct MultiScaleDilatedConv {
    /// Convolution layers with different dilation rates
    convs: Vec<DilatedConv1D>,
}

impl MultiScaleDilatedConv {
    /// Create a new multi-scale dilated convolution block
    ///
    /// # Arguments
    /// - `in_channels` - Number of input channels
    /// - `out_channels_per_scale` - Output channels for each scale
    /// - `kernel_size` - Size of convolution kernel
    /// - `dilation_rates` - Dilation rates for each scale
    pub fn new(
        in_channels: usize,
        out_channels_per_scale: usize,
        kernel_size: usize,
        dilation_rates: &[usize],
    ) -> Self {
        let convs = dilation_rates
            .iter()
            .map(|&d| DilatedConv1D::new(in_channels, out_channels_per_scale, kernel_size, d))
            .collect();

        Self { convs }
    }

    /// Total output channels
    pub fn out_channels(&self) -> usize {
        self.convs.len() * self.convs.first().map(|c| c.out_channels).unwrap_or(0)
    }

    /// Apply multi-scale convolution
    ///
    /// # Arguments
    /// - `input` - Input tensor of shape (in_channels, sequence_length)
    ///
    /// # Returns
    /// - Output tensor of shape (total_out_channels, sequence_length)
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let outputs: Vec<Array2<f64>> = self.convs.iter().map(|c| c.forward(input)).collect();

        // Concatenate along channel dimension
        let total_channels = outputs.iter().map(|o| o.dim().0).sum();
        let seq_len = outputs.first().map(|o| o.dim().1).unwrap_or(0);

        let mut result = Array2::zeros((total_channels, seq_len));
        let mut offset = 0;

        for output in &outputs {
            let n_channels = output.dim().0;
            result
                .slice_mut(ndarray::s![offset..offset + n_channels, ..])
                .assign(output);
            offset += n_channels;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_dilated_conv_creation() {
        let conv = DilatedConv1D::new(5, 32, 3, 4);
        assert_eq!(conv.in_channels, 5);
        assert_eq!(conv.out_channels, 32);
        assert_eq!(conv.kernel_size, 3);
        assert_eq!(conv.dilation, 4);
    }

    #[test]
    fn test_receptive_field() {
        let conv = DilatedConv1D::new(5, 32, 3, 4);
        // RF = (kernel_size - 1) * dilation + 1 = (3-1)*4 + 1 = 9
        assert_eq!(conv.receptive_field(), 9);
    }

    #[test]
    fn test_forward_shape() {
        let conv = DilatedConv1D::new(3, 8, 3, 2);
        let input = Array2::zeros((3, 100));
        let output = conv.forward(&input);
        assert_eq!(output.dim(), (8, 100));
    }

    #[test]
    fn test_causal() {
        // Test that output at time t only depends on inputs at times <= t
        let weights = vec![
            Array2::ones((1, 3)), // Simple sum kernel
        ];
        let bias = Array1::zeros(1);
        let conv = DilatedConv1D::with_weights(1, 1, 3, 1, weights, bias);

        // Input: [0, 0, 0, 1, 0, 0, 0]
        let mut input = Array2::zeros((1, 7));
        input[[0, 3]] = 1.0;

        let output = conv.forward(&input);

        // Output at t=3 should include the impulse
        // But output at t=0,1,2 should not
        assert_eq!(output[[0, 0]], 0.0);
        assert_eq!(output[[0, 1]], 0.0);
        assert_eq!(output[[0, 2]], 0.0);
        assert!(output[[0, 3]] > 0.0);
    }

    #[test]
    fn test_multi_scale() {
        let block = MultiScaleDilatedConv::new(5, 16, 3, &[1, 2, 4, 8]);
        assert_eq!(block.out_channels(), 64); // 4 scales * 16 channels

        let input = Array2::zeros((5, 50));
        let output = block.forward(&input);
        assert_eq!(output.dim(), (64, 50));
    }
}
