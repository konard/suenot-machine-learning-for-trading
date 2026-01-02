//! Residual blocks for ResNet

use super::layers::{BatchNorm1d, Conv1d, ReLU};
use ndarray::Array3;
use serde::{Deserialize, Serialize};

/// Basic residual block with two 3x3 convolutions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicBlock {
    /// First convolution
    pub conv1: Conv1d,
    /// First batch norm
    pub bn1: BatchNorm1d,
    /// Second convolution
    pub conv2: Conv1d,
    /// Second batch norm
    pub bn2: BatchNorm1d,
    /// ReLU activation
    pub relu: ReLU,
    /// Downsample projection for skip connection (if needed)
    pub downsample: Option<(Conv1d, BatchNorm1d)>,
    /// Stride
    pub stride: usize,
}

impl BasicBlock {
    /// Expansion factor for this block type
    pub const EXPANSION: usize = 1;

    /// Create a new BasicBlock
    ///
    /// # Arguments
    ///
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `stride` - Stride for the first convolution (for downsampling)
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let conv1 = Conv1d::new(in_channels, out_channels, 3, stride, 1, false);
        let bn1 = BatchNorm1d::new(out_channels);
        let conv2 = Conv1d::new(out_channels, out_channels, 3, 1, 1, false);
        let bn2 = BatchNorm1d::new(out_channels);
        let relu = ReLU::new();

        // Create downsample layer if dimensions change
        let downsample = if stride != 1 || in_channels != out_channels * Self::EXPANSION {
            Some((
                Conv1d::new(in_channels, out_channels * Self::EXPANSION, 1, stride, 0, false),
                BatchNorm1d::new(out_channels * Self::EXPANSION),
            ))
        } else {
            None
        };

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            relu,
            downsample,
            stride,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &Array3<f32>) -> Array3<f32> {
        // Save input for skip connection
        let identity = x.clone();

        // Main path
        let mut out = self.conv1.forward(x);
        out = self.bn1.forward(&out);
        out = self.relu.forward(&out);

        out = self.conv2.forward(&out);
        out = self.bn2.forward(&out);

        // Skip connection
        let identity = match &self.downsample {
            Some((conv, bn)) => {
                let projected = conv.forward(&identity);
                bn.forward(&projected)
            }
            None => identity,
        };

        // Add skip connection
        out = &out + &identity;
        out = self.relu.forward(&out);

        out
    }

    /// Get number of parameters
    pub fn num_params(&self) -> usize {
        let mut params = self.conv1.num_params()
            + self.conv2.num_params()
            + self.bn1.num_features * 2
            + self.bn2.num_features * 2;

        if let Some((conv, bn)) = &self.downsample {
            params += conv.num_params() + bn.num_features * 2;
        }

        params
    }
}

/// Bottleneck residual block with 1x1, 3x3, 1x1 convolutions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckBlock {
    /// First 1x1 convolution (reduce)
    pub conv1: Conv1d,
    /// First batch norm
    pub bn1: BatchNorm1d,
    /// Second 3x3 convolution (process)
    pub conv2: Conv1d,
    /// Second batch norm
    pub bn2: BatchNorm1d,
    /// Third 1x1 convolution (expand)
    pub conv3: Conv1d,
    /// Third batch norm
    pub bn3: BatchNorm1d,
    /// ReLU activation
    pub relu: ReLU,
    /// Downsample projection for skip connection
    pub downsample: Option<(Conv1d, BatchNorm1d)>,
    /// Stride
    pub stride: usize,
}

impl BottleneckBlock {
    /// Expansion factor for this block type
    pub const EXPANSION: usize = 4;

    /// Create a new BottleneckBlock
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        // 1x1 reduce
        let conv1 = Conv1d::new(in_channels, out_channels, 1, 1, 0, false);
        let bn1 = BatchNorm1d::new(out_channels);

        // 3x3 process
        let conv2 = Conv1d::new(out_channels, out_channels, 3, stride, 1, false);
        let bn2 = BatchNorm1d::new(out_channels);

        // 1x1 expand
        let conv3 = Conv1d::new(out_channels, out_channels * Self::EXPANSION, 1, 1, 0, false);
        let bn3 = BatchNorm1d::new(out_channels * Self::EXPANSION);

        let relu = ReLU::new();

        // Create downsample layer if dimensions change
        let downsample = if stride != 1 || in_channels != out_channels * Self::EXPANSION {
            Some((
                Conv1d::new(in_channels, out_channels * Self::EXPANSION, 1, stride, 0, false),
                BatchNorm1d::new(out_channels * Self::EXPANSION),
            ))
        } else {
            None
        };

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            relu,
            downsample,
            stride,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &Array3<f32>) -> Array3<f32> {
        let identity = x.clone();

        // 1x1 reduce
        let mut out = self.conv1.forward(x);
        out = self.bn1.forward(&out);
        out = self.relu.forward(&out);

        // 3x3 process
        out = self.conv2.forward(&out);
        out = self.bn2.forward(&out);
        out = self.relu.forward(&out);

        // 1x1 expand
        out = self.conv3.forward(&out);
        out = self.bn3.forward(&out);

        // Skip connection
        let identity = match &self.downsample {
            Some((conv, bn)) => {
                let projected = conv.forward(&identity);
                bn.forward(&projected)
            }
            None => identity,
        };

        // Add skip connection
        out = &out + &identity;
        out = self.relu.forward(&out);

        out
    }

    /// Get number of parameters
    pub fn num_params(&self) -> usize {
        let mut params = self.conv1.num_params()
            + self.conv2.num_params()
            + self.conv3.num_params()
            + self.bn1.num_features * 2
            + self.bn2.num_features * 2
            + self.bn3.num_features * 2;

        if let Some((conv, bn)) = &self.downsample {
            params += conv.num_params() + bn.num_features * 2;
        }

        params
    }
}

/// Generic residual block trait
pub trait ResidualBlock {
    /// Expansion factor
    const EXPANSION: usize;

    /// Forward pass
    fn forward(&self, x: &Array3<f32>) -> Array3<f32>;

    /// Get number of parameters
    fn num_params(&self) -> usize;
}

impl ResidualBlock for BasicBlock {
    const EXPANSION: usize = 1;

    fn forward(&self, x: &Array3<f32>) -> Array3<f32> {
        BasicBlock::forward(self, x)
    }

    fn num_params(&self) -> usize {
        BasicBlock::num_params(self)
    }
}

impl ResidualBlock for BottleneckBlock {
    const EXPANSION: usize = 4;

    fn forward(&self, x: &Array3<f32>) -> Array3<f32> {
        BottleneckBlock::forward(self, x)
    }

    fn num_params(&self) -> usize {
        BottleneckBlock::num_params(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_block_same_dims() {
        let block = BasicBlock::new(64, 64, 1);
        let input = Array3::ones((2, 64, 32));
        let output = block.forward(&input);
        assert_eq!(output.shape(), &[2, 64, 32]);
    }

    #[test]
    fn test_basic_block_downsample() {
        let block = BasicBlock::new(64, 128, 2);
        let input = Array3::ones((2, 64, 32));
        let output = block.forward(&input);
        assert_eq!(output.shape(), &[2, 128, 16]);
    }

    #[test]
    fn test_bottleneck_block() {
        let block = BottleneckBlock::new(64, 64, 1);
        let input = Array3::ones((2, 64, 32));
        let output = block.forward(&input);
        assert_eq!(output.shape(), &[2, 256, 32]); // 64 * 4 = 256
    }
}
