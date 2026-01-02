//! TCN Residual Block Implementation

use ndarray::Array2;

use super::layer::CausalConv1d;

/// Residual block for TCN
///
/// Structure:
/// ```text
/// Input
///   |
///   +---> CausalConv1d --> ReLU --> Dropout --> CausalConv1d --> ReLU --> Dropout
///   |                                                                         |
///   +--------------------------(residual connection)--------------------------+
///   |
///   v
/// Output
/// ```
#[derive(Debug, Clone)]
pub struct TCNResidualBlock {
    /// First convolution layer
    pub conv1: CausalConv1d,
    /// Second convolution layer
    pub conv2: CausalConv1d,
    /// Optional 1x1 convolution for matching dimensions
    pub residual_conv: Option<CausalConv1d>,
    /// Dropout probability
    pub dropout_prob: f64,
    /// Dilation factor for this block
    pub dilation: usize,
}

impl TCNResidualBlock {
    /// Create a new residual block
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Convolution kernel size
    /// * `dilation` - Dilation factor
    /// * `dropout` - Dropout probability
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        dropout: f64,
    ) -> Self {
        let conv1 = CausalConv1d::new(in_channels, out_channels, kernel_size, dilation);
        let conv2 = CausalConv1d::new(out_channels, out_channels, kernel_size, dilation);

        // If channels don't match, we need a 1x1 conv for the residual connection
        let residual_conv = if in_channels != out_channels {
            Some(CausalConv1d::new(in_channels, out_channels, 1, 1))
        } else {
            None
        };

        Self {
            conv1,
            conv2,
            residual_conv,
            dropout_prob: dropout,
            dilation,
        }
    }

    /// ReLU activation function
    fn relu(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| v.max(0.0))
    }

    /// Apply dropout (simplified - in training mode would randomly zero elements)
    fn dropout(&self, x: &Array2<f64>, training: bool) -> Array2<f64> {
        if training && self.dropout_prob > 0.0 {
            // Simplified dropout - in practice, randomly zero elements
            x * (1.0 - self.dropout_prob)
        } else {
            x.clone()
        }
    }

    /// Forward pass through the residual block
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [channels, seq_len]
    /// * `training` - Whether in training mode (affects dropout)
    ///
    /// # Returns
    /// Output tensor of shape [out_channels, seq_len]
    pub fn forward(&self, input: &Array2<f64>, training: bool) -> Array2<f64> {
        // First convolution block
        let out = self.conv1.forward(input);
        let out = Self::relu(&out);
        let out = self.dropout(&out, training);

        // Second convolution block
        let out = self.conv2.forward(&out);
        let out = Self::relu(&out);
        let out = self.dropout(&out, training);

        // Residual connection
        let residual = match &self.residual_conv {
            Some(conv) => conv.forward(input),
            None => input.clone(),
        };

        // Add residual and apply final ReLU
        Self::relu(&(&out + &residual))
    }

    /// Get receptive field of this block
    pub fn receptive_field(&self) -> usize {
        // Two convolutions with same dilation
        2 * (self.conv1.kernel_size - 1) * self.dilation + 1
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        let mut params = self.conv1.num_parameters() + self.conv2.num_parameters();
        if let Some(ref conv) = self.residual_conv {
            params += conv.num_parameters();
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residual_block_creation() {
        let block = TCNResidualBlock::new(16, 32, 3, 1, 0.2);
        assert!(block.residual_conv.is_some()); // Different channels

        let block2 = TCNResidualBlock::new(32, 32, 3, 1, 0.2);
        assert!(block2.residual_conv.is_none()); // Same channels
    }

    #[test]
    fn test_residual_block_forward() {
        let block = TCNResidualBlock::new(4, 8, 3, 1, 0.0);
        let input = Array2::ones((4, 20));
        let output = block.forward(&input, false);

        assert_eq!(output.dim(), (8, 20));
    }

    #[test]
    fn test_output_non_negative() {
        // Due to ReLU, output should be non-negative
        let block = TCNResidualBlock::new(4, 4, 3, 2, 0.0);
        let input = Array2::from_elem((4, 20), -1.0);
        let output = block.forward(&input, false);

        for &val in output.iter() {
            assert!(val >= 0.0, "Output should be non-negative due to ReLU");
        }
    }
}
