//! Depthwise Separable 1D Convolution
//!
//! Combines depthwise and pointwise convolutions for efficient computation.
//! Achieves similar accuracy to standard convolution with 8-9x fewer operations.

use ndarray::{Array1, Array2, Array3, Axis};

use super::{
    depthwise::DepthwiseConv1d,
    pointwise::PointwiseConv1d,
    Activation, BatchNorm1d, ConvError, Padding,
};

/// Depthwise Separable 1D Convolution
///
/// Factorizes standard convolution into:
/// 1. Depthwise: separate filter per input channel
/// 2. Pointwise: 1x1 conv to mix channels
///
/// Computational reduction: ~8-9x fewer operations
///
/// # Example
///
/// ```rust
/// use dsc_trading::convolution::DepthwiseSeparableConv1d;
/// use ndarray::Array2;
///
/// let conv = DepthwiseSeparableConv1d::new(10, 64, 3).unwrap();
/// let input = Array2::zeros((10, 100));
/// let output = conv.forward(&input);
/// assert_eq!(output.dim(), (64, 100));
/// ```
#[derive(Debug, Clone)]
pub struct DepthwiseSeparableConv1d {
    /// Depthwise convolution layer
    pub depthwise: DepthwiseConv1d,
    /// Batch normalization after depthwise
    pub bn1: BatchNorm1d,
    /// Pointwise convolution layer
    pub pointwise: PointwiseConv1d,
    /// Batch normalization after pointwise
    pub bn2: BatchNorm1d,
    /// Use batch normalization
    pub use_batchnorm: bool,
    /// Residual connection (if in_channels == out_channels)
    pub use_residual: bool,
}

impl DepthwiseSeparableConv1d {
    /// Create a new depthwise separable convolution layer
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the depthwise convolution kernel
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
    ) -> Result<Self, ConvError> {
        let depthwise = DepthwiseConv1d::new(in_channels, kernel_size)?
            .with_padding(Padding::Same)
            .with_activation(Activation::ReLU);

        let pointwise = PointwiseConv1d::new(in_channels, out_channels)?
            .with_activation(Activation::ReLU);

        let bn1 = BatchNorm1d::new(in_channels);
        let bn2 = BatchNorm1d::new(out_channels);

        Ok(Self {
            depthwise,
            bn1,
            pointwise,
            bn2,
            use_batchnorm: true,
            use_residual: in_channels == out_channels,
        })
    }

    /// Create with custom configuration
    pub fn with_config(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
    ) -> Result<Self, ConvError> {
        let depthwise = DepthwiseConv1d::new(in_channels, kernel_size)?
            .with_stride(stride)
            .with_dilation(dilation)
            .with_padding(Padding::Same)
            .with_activation(Activation::ReLU);

        let pointwise = PointwiseConv1d::new(in_channels, out_channels)?
            .with_activation(Activation::ReLU);

        let bn1 = BatchNorm1d::new(in_channels);
        let bn2 = BatchNorm1d::new(out_channels);

        Ok(Self {
            depthwise,
            bn1,
            pointwise,
            bn2,
            use_batchnorm: true,
            use_residual: in_channels == out_channels && stride == 1,
        })
    }

    /// Builder: disable batch normalization
    pub fn without_batchnorm(mut self) -> Self {
        self.use_batchnorm = false;
        self
    }

    /// Builder: disable residual connection
    pub fn without_residual(mut self) -> Self {
        self.use_residual = false;
        self
    }

    /// Builder: set custom activation for depthwise
    pub fn with_depthwise_activation(mut self, activation: Activation) -> Self {
        self.depthwise.activation = activation;
        self
    }

    /// Builder: set custom activation for pointwise
    pub fn with_pointwise_activation(mut self, activation: Activation) -> Self {
        self.pointwise.activation = activation;
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
        // Depthwise convolution
        let mut x = self.depthwise.forward(input);

        // Batch normalization after depthwise
        if self.use_batchnorm {
            x = self.bn1.forward(&x);
        }

        // Pointwise convolution
        let mut output = self.pointwise.forward(&x);

        // Batch normalization after pointwise
        if self.use_batchnorm {
            output = self.bn2.forward(&output);
        }

        // Residual connection
        if self.use_residual && input.dim() == output.dim() {
            for ((i, j), val) in output.indexed_iter_mut() {
                *val += input[[i, j]];
            }
        }

        output
    }

    /// Forward pass for batch input
    pub fn forward_batch(&self, input: &Array3<f64>) -> Array3<f64> {
        let (batch_size, _, seq_len) = input.dim();
        let out_len = self.depthwise.output_length(seq_len);

        let mut output = Array3::zeros((batch_size, self.pointwise.out_channels, out_len));

        for b in 0..batch_size {
            let input_slice = input.index_axis(Axis(0), b).to_owned();
            let output_slice = self.forward(&input_slice);

            for c in 0..self.pointwise.out_channels {
                for i in 0..out_len {
                    output[[b, c, i]] = output_slice[[c, i]];
                }
            }
        }

        output
    }

    /// Count number of parameters
    pub fn num_parameters(&self) -> usize {
        self.depthwise.num_parameters() + self.pointwise.num_parameters()
    }

    /// Count number of FLOPs for given input length
    pub fn flops(&self, input_length: usize) -> usize {
        let dw_out_len = self.depthwise.output_length(input_length);
        self.depthwise.flops(input_length) + self.pointwise.flops(dw_out_len)
    }

    /// Compare FLOPs with standard convolution
    pub fn flops_reduction(&self, input_length: usize) -> f64 {
        let k = self.depthwise.kernel_size;
        let c_in = self.depthwise.in_channels;
        let c_out = self.pointwise.out_channels;

        // Standard conv FLOPs: K * C_in * C_out * L * 2
        let standard_flops = k * c_in * c_out * input_length * 2;

        // DSC FLOPs
        let dsc_flops = self.flops(input_length);

        standard_flops as f64 / dsc_flops as f64
    }
}

/// Stack of depthwise separable convolution blocks
#[derive(Debug, Clone)]
pub struct DSCStack {
    /// List of DSC blocks
    pub blocks: Vec<DepthwiseSeparableConv1d>,
    /// Final projection layer
    pub output_proj: Option<PointwiseConv1d>,
}

impl DSCStack {
    /// Create a stack of DSC blocks
    ///
    /// # Arguments
    /// * `in_channels` - Input channels
    /// * `hidden_channels` - Hidden layer channels
    /// * `out_channels` - Output channels
    /// * `num_blocks` - Number of DSC blocks
    /// * `kernel_size` - Kernel size for all blocks
    pub fn new(
        in_channels: usize,
        hidden_channels: usize,
        out_channels: usize,
        num_blocks: usize,
        kernel_size: usize,
    ) -> Result<Self, ConvError> {
        let mut blocks = Vec::with_capacity(num_blocks);

        // First block: in_channels -> hidden_channels
        blocks.push(DepthwiseSeparableConv1d::new(
            in_channels,
            hidden_channels,
            kernel_size,
        )?);

        // Middle blocks: hidden_channels -> hidden_channels
        for _ in 1..num_blocks {
            blocks.push(DepthwiseSeparableConv1d::new(
                hidden_channels,
                hidden_channels,
                kernel_size,
            )?);
        }

        // Output projection if needed
        let output_proj = if hidden_channels != out_channels {
            Some(PointwiseConv1d::new(hidden_channels, out_channels)?)
        } else {
            None
        };

        Ok(Self { blocks, output_proj })
    }

    /// Create with dilated convolutions (for larger receptive field)
    pub fn with_dilation(
        in_channels: usize,
        hidden_channels: usize,
        out_channels: usize,
        num_blocks: usize,
        kernel_size: usize,
    ) -> Result<Self, ConvError> {
        let mut blocks = Vec::with_capacity(num_blocks);

        // First block
        blocks.push(DepthwiseSeparableConv1d::with_config(
            in_channels,
            hidden_channels,
            kernel_size,
            1,
            1, // dilation = 1
        )?);

        // Remaining blocks with exponentially increasing dilation
        for i in 1..num_blocks {
            let dilation = 2usize.pow(i as u32);
            blocks.push(DepthwiseSeparableConv1d::with_config(
                hidden_channels,
                hidden_channels,
                kernel_size,
                1,
                dilation,
            )?);
        }

        let output_proj = if hidden_channels != out_channels {
            Some(PointwiseConv1d::new(hidden_channels, out_channels)?)
        } else {
            None
        };

        Ok(Self { blocks, output_proj })
    }

    /// Forward pass
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let mut x = input.clone();

        for block in &self.blocks {
            x = block.forward(&x);
        }

        if let Some(ref proj) = self.output_proj {
            x = proj.forward(&x);
        }

        x
    }

    /// Forward pass with skip connections
    pub fn forward_with_skips(&self, input: &Array2<f64>) -> (Array2<f64>, Vec<Array2<f64>>) {
        let mut x = input.clone();
        let mut skips = Vec::with_capacity(self.blocks.len());

        for block in &self.blocks {
            x = block.forward(&x);
            skips.push(x.clone());
        }

        if let Some(ref proj) = self.output_proj {
            x = proj.forward(&x);
        }

        (x, skips)
    }

    /// Count total parameters
    pub fn num_parameters(&self) -> usize {
        let block_params: usize = self.blocks.iter().map(|b| b.num_parameters()).sum();
        let proj_params = self.output_proj.as_ref().map_or(0, |p| p.num_parameters());
        block_params + proj_params
    }

    /// Receptive field size
    pub fn receptive_field(&self) -> usize {
        let mut rf = 1;
        for block in &self.blocks {
            let k = block.depthwise.kernel_size;
            let d = block.depthwise.dilation;
            rf += (k - 1) * d;
        }
        rf
    }
}

/// Mobile-style inverted residual block
#[derive(Debug, Clone)]
pub struct InvertedResidual {
    /// Expansion layer (1x1)
    pub expand: Option<PointwiseConv1d>,
    /// Depthwise convolution
    pub depthwise: DepthwiseConv1d,
    /// Projection layer (1x1)
    pub project: PointwiseConv1d,
    /// Use residual connection
    pub use_residual: bool,
}

impl InvertedResidual {
    /// Create inverted residual block
    ///
    /// # Arguments
    /// * `in_channels` - Input channels
    /// * `out_channels` - Output channels
    /// * `kernel_size` - Depthwise kernel size
    /// * `expansion_factor` - Channel expansion factor (e.g., 6 for MobileNetV2)
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        expansion_factor: usize,
    ) -> Result<Self, ConvError> {
        let hidden_channels = in_channels * expansion_factor;

        // Expansion (skip if expansion_factor == 1)
        let expand = if expansion_factor > 1 {
            Some(
                PointwiseConv1d::new(in_channels, hidden_channels)?
                    .with_activation(Activation::ReLU),
            )
        } else {
            None
        };

        // Depthwise
        let dw_in = if expansion_factor > 1 {
            hidden_channels
        } else {
            in_channels
        };
        let depthwise = DepthwiseConv1d::new(dw_in, kernel_size)?
            .with_padding(Padding::Same)
            .with_activation(Activation::ReLU);

        // Projection (linear activation for residual)
        let project =
            PointwiseConv1d::new(dw_in, out_channels)?.with_activation(Activation::None);

        Ok(Self {
            expand,
            depthwise,
            project,
            use_residual: in_channels == out_channels,
        })
    }

    /// Forward pass
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let x = if let Some(ref expand) = self.expand {
            expand.forward(input)
        } else {
            input.clone()
        };

        let x = self.depthwise.forward(&x);
        let mut output = self.project.forward(&x);

        if self.use_residual && input.dim() == output.dim() {
            for ((i, j), val) in output.indexed_iter_mut() {
                *val += input[[i, j]];
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dsc_creation() {
        let conv = DepthwiseSeparableConv1d::new(10, 64, 3).unwrap();
        assert_eq!(conv.depthwise.in_channels, 10);
        assert_eq!(conv.pointwise.out_channels, 64);
    }

    #[test]
    fn test_dsc_forward() {
        let conv = DepthwiseSeparableConv1d::new(10, 64, 3).unwrap();
        let input = Array2::ones((10, 100));
        let output = conv.forward(&input);

        assert_eq!(output.dim(), (64, 100));
    }

    #[test]
    fn test_dsc_parameter_reduction() {
        // Standard conv: 10 * 64 * 3 = 1920 + 64 bias = 1984
        // DSC: depthwise(10*3 + 10) + pointwise(10*64 + 64) = 40 + 704 = 744
        let conv = DepthwiseSeparableConv1d::new(10, 64, 3).unwrap();
        let params = conv.num_parameters();

        // Should be significantly less than standard conv
        assert!(params < 1984);
        println!("DSC params: {}, Standard params: 1984", params);
    }

    #[test]
    fn test_dsc_flops_reduction() {
        let conv = DepthwiseSeparableConv1d::new(64, 64, 3).unwrap();
        let reduction = conv.flops_reduction(100);

        // Should be around 8-9x reduction
        assert!(reduction > 5.0);
        println!("FLOPs reduction factor: {:.2}x", reduction);
    }

    #[test]
    fn test_dsc_stack() {
        let stack = DSCStack::new(10, 64, 32, 4, 3).unwrap();
        let input = Array2::ones((10, 100));
        let output = stack.forward(&input);

        assert_eq!(output.dim(), (32, 100));
    }

    #[test]
    fn test_dsc_stack_receptive_field() {
        let stack = DSCStack::with_dilation(10, 64, 32, 4, 3).unwrap();
        let rf = stack.receptive_field();

        // RF = 1 + (3-1)*1 + (3-1)*2 + (3-1)*4 + (3-1)*8 = 1 + 2 + 4 + 8 + 16 = 31
        println!("Receptive field: {}", rf);
        assert!(rf > 20);
    }

    #[test]
    fn test_inverted_residual() {
        let block = InvertedResidual::new(32, 32, 3, 6).unwrap();
        let input = Array2::ones((32, 100));
        let output = block.forward(&input);

        assert_eq!(output.dim(), (32, 100));
    }
}
