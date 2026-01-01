//! WaveNet-style Architecture Implementation
//!
//! This module implements the core components of the WaveNet architecture:
//! - Gated activation units
//! - Residual connections
//! - Skip connections
//! - Stacked dilated convolutions with exponentially increasing dilation

use ndarray::{Array1, Array2};

use super::causal::Conv1x1;
use super::dilated::DilatedConv1D;

/// Gated Activation Unit
///
/// Implements the gated activation from WaveNet:
/// z = tanh(Wf * x) ⊙ σ(Wg * x)
///
/// where Wf is the "filter" and Wg is the "gate"
#[derive(Debug, Clone)]
pub struct GatedActivation {
    /// Filter convolution
    filter_conv: DilatedConv1D,
    /// Gate convolution
    gate_conv: DilatedConv1D,
}

impl GatedActivation {
    /// Create a new gated activation unit
    pub fn new(channels: usize, kernel_size: usize, dilation: usize) -> Self {
        Self {
            filter_conv: DilatedConv1D::new(channels, channels, kernel_size, dilation),
            gate_conv: DilatedConv1D::new(channels, channels, kernel_size, dilation),
        }
    }

    /// Apply the gated activation
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let filter_out = self.filter_conv.forward(input);
        let gate_out = self.gate_conv.forward(input);

        // tanh(filter) * sigmoid(gate)
        let filter_activated = DilatedConv1D::tanh(&filter_out);
        let gate_activated = DilatedConv1D::sigmoid(&gate_out);

        // Element-wise multiplication
        &filter_activated * &gate_activated
    }
}

/// Dilated Residual Block
///
/// A single residual block with:
/// - Gated activation
/// - 1x1 convolution for residual path
/// - 1x1 convolution for skip connection
#[derive(Debug, Clone)]
pub struct DilatedResidualBlock {
    /// Gated activation unit
    gated_activation: GatedActivation,
    /// 1x1 conv for residual connection
    residual_conv: Conv1x1,
    /// 1x1 conv for skip connection
    skip_conv: Conv1x1,
    /// Dilation rate
    dilation: usize,
}

impl DilatedResidualBlock {
    /// Create a new residual block
    ///
    /// # Arguments
    /// - `residual_channels` - Number of channels in the residual path
    /// - `skip_channels` - Number of channels in the skip path
    /// - `kernel_size` - Size of convolution kernel
    /// - `dilation` - Dilation rate
    pub fn new(
        residual_channels: usize,
        skip_channels: usize,
        kernel_size: usize,
        dilation: usize,
    ) -> Self {
        Self {
            gated_activation: GatedActivation::new(residual_channels, kernel_size, dilation),
            residual_conv: Conv1x1::new(residual_channels, residual_channels),
            skip_conv: Conv1x1::new(residual_channels, skip_channels),
            dilation,
        }
    }

    /// Get the dilation rate
    pub fn dilation(&self) -> usize {
        self.dilation
    }

    /// Apply the residual block
    ///
    /// # Returns
    /// - (residual_output, skip_output)
    pub fn forward(&self, input: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        // Gated activation
        let activated = self.gated_activation.forward(input);

        // Skip connection output
        let skip = self.skip_conv.forward(&activated);

        // Residual connection: input + conv(activated)
        let residual = self.residual_conv.forward(&activated) + input;

        (residual, skip)
    }
}

/// Dilated Convolution Stack (WaveNet-style)
///
/// A stack of dilated residual blocks with:
/// - Input convolution to project to residual channels
/// - Multiple residual blocks with increasing dilation
/// - Skip connections from all blocks
/// - Output convolutions
#[derive(Debug, Clone)]
pub struct DilatedConvStack {
    /// Input projection
    input_conv: Conv1x1,
    /// Residual blocks
    blocks: Vec<DilatedResidualBlock>,
    /// Output convolution 1
    output_conv1: Conv1x1,
    /// Output convolution 2
    output_conv2: Conv1x1,
    /// Skip channels
    skip_channels: usize,
}

impl DilatedConvStack {
    /// Create a new dilated convolution stack
    ///
    /// # Arguments
    /// - `input_channels` - Number of input features
    /// - `residual_channels` - Number of channels in residual path
    /// - `dilation_rates` - Dilation rates for each block (e.g., [1, 2, 4, 8, 16, 32])
    pub fn new(
        input_channels: usize,
        residual_channels: usize,
        dilation_rates: &[usize],
    ) -> Self {
        let skip_channels = residual_channels * 2;
        let kernel_size = 3;

        let blocks = dilation_rates
            .iter()
            .map(|&d| {
                DilatedResidualBlock::new(residual_channels, skip_channels, kernel_size, d)
            })
            .collect();

        Self {
            input_conv: Conv1x1::new(input_channels, residual_channels),
            blocks,
            output_conv1: Conv1x1::new(skip_channels, residual_channels),
            output_conv2: Conv1x1::new(residual_channels, 3), // 3 outputs: direction, magnitude, volatility
            skip_channels,
        }
    }

    /// Create with custom output size
    pub fn with_output_size(
        input_channels: usize,
        residual_channels: usize,
        dilation_rates: &[usize],
        output_size: usize,
    ) -> Self {
        let skip_channels = residual_channels * 2;
        let kernel_size = 3;

        let blocks = dilation_rates
            .iter()
            .map(|&d| {
                DilatedResidualBlock::new(residual_channels, skip_channels, kernel_size, d)
            })
            .collect();

        Self {
            input_conv: Conv1x1::new(input_channels, residual_channels),
            blocks,
            output_conv1: Conv1x1::new(skip_channels, residual_channels),
            output_conv2: Conv1x1::new(residual_channels, output_size),
            skip_channels,
        }
    }

    /// Get the total receptive field
    pub fn receptive_field(&self) -> usize {
        let kernel_size = 3;
        let dilation_sum: usize = self.blocks.iter().map(|b| b.dilation()).sum();
        1 + (kernel_size - 1) * dilation_sum
    }

    /// Get the number of blocks
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Apply the full model
    ///
    /// # Arguments
    /// - `input` - Input tensor of shape (input_channels, sequence_length)
    ///
    /// # Returns
    /// - Output tensor of shape (output_channels, sequence_length)
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        // Input projection
        let mut x = self.input_conv.forward(input);

        // Collect skip connections
        let mut skip_sum = Array2::zeros((self.skip_channels, input.dim().1));

        // Process through residual blocks
        for block in &self.blocks {
            let (residual, skip) = block.forward(&x);
            x = residual;
            skip_sum = skip_sum + skip;
        }

        // Apply ReLU to skip sum
        let activated = DilatedConv1D::relu(&skip_sum);

        // First output convolution
        let out = self.output_conv1.forward(&activated);
        let out = DilatedConv1D::relu(&out);

        // Final output convolution
        self.output_conv2.forward(&out)
    }

    /// Get prediction for the last timestep only
    pub fn predict_last(&self, input: &Array2<f64>) -> Array1<f64> {
        let output = self.forward(input);
        let seq_len = output.dim().1;
        output.column(seq_len - 1).to_owned()
    }

    /// Extract multi-scale features (intermediate skip connections)
    pub fn extract_features(&self, input: &Array2<f64>) -> Vec<Array2<f64>> {
        let mut x = self.input_conv.forward(input);
        let mut features = Vec::new();

        for block in &self.blocks {
            let (residual, skip) = block.forward(&x);
            x = residual;
            features.push(skip);
        }

        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gated_activation_shape() {
        let gated = GatedActivation::new(32, 3, 4);
        let input = Array2::zeros((32, 100));
        let output = gated.forward(&input);
        assert_eq!(output.dim(), (32, 100));
    }

    #[test]
    fn test_residual_block_shape() {
        let block = DilatedResidualBlock::new(32, 64, 3, 8);
        let input = Array2::zeros((32, 100));
        let (residual, skip) = block.forward(&input);
        assert_eq!(residual.dim(), (32, 100));
        assert_eq!(skip.dim(), (64, 100));
    }

    #[test]
    fn test_wavenet_stack_shape() {
        let stack = DilatedConvStack::new(5, 32, &[1, 2, 4, 8, 16, 32]);
        let input = Array2::zeros((5, 100));
        let output = stack.forward(&input);
        assert_eq!(output.dim(), (3, 100)); // Default 3 outputs
    }

    #[test]
    fn test_wavenet_receptive_field() {
        let stack = DilatedConvStack::new(5, 32, &[1, 2, 4, 8, 16, 32]);
        // RF = 1 + (kernel_size - 1) * sum(dilations)
        // RF = 1 + 2 * (1+2+4+8+16+32) = 1 + 2*63 = 127
        assert_eq!(stack.receptive_field(), 127);
    }

    #[test]
    fn test_feature_extraction() {
        let stack = DilatedConvStack::new(5, 32, &[1, 2, 4, 8]);
        let input = Array2::zeros((5, 100));
        let features = stack.extract_features(&input);
        assert_eq!(features.len(), 4); // One per block
    }

    #[test]
    fn test_predict_last() {
        let stack = DilatedConvStack::new(5, 32, &[1, 2, 4]);
        let input = Array2::zeros((5, 50));
        let prediction = stack.predict_last(&input);
        assert_eq!(prediction.len(), 3);
    }
}
