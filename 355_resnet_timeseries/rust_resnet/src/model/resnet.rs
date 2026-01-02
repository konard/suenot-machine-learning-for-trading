//! ResNet architecture for time series

use super::blocks::{BasicBlock, BottleneckBlock};
use super::layers::{AdaptiveAvgPool1d, BatchNorm1d, Conv1d, Linear, MaxPool1d, ReLU};
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

/// ResNet-18 model for time series classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResNet18 {
    /// Initial convolution
    pub conv1: Conv1d,
    /// Initial batch norm
    pub bn1: BatchNorm1d,
    /// ReLU activation
    pub relu: ReLU,
    /// Max pooling
    pub maxpool: MaxPool1d,
    /// Layer 1 (2 blocks, 64 channels)
    pub layer1: Vec<BasicBlock>,
    /// Layer 2 (2 blocks, 128 channels)
    pub layer2: Vec<BasicBlock>,
    /// Layer 3 (2 blocks, 256 channels)
    pub layer3: Vec<BasicBlock>,
    /// Layer 4 (2 blocks, 512 channels)
    pub layer4: Vec<BasicBlock>,
    /// Adaptive average pooling
    pub avgpool: AdaptiveAvgPool1d,
    /// Final fully connected layer
    pub fc: Linear,
    /// Number of input channels
    pub in_channels: usize,
    /// Number of output classes
    pub num_classes: usize,
}

impl ResNet18 {
    /// Create a new ResNet-18 model
    ///
    /// # Arguments
    ///
    /// * `in_channels` - Number of input channels (features)
    /// * `num_classes` - Number of output classes
    pub fn new(in_channels: usize, num_classes: usize) -> Self {
        // Initial layers
        let conv1 = Conv1d::new(in_channels, 64, 7, 2, 3, false);
        let bn1 = BatchNorm1d::new(64);
        let relu = ReLU::new();
        let maxpool = MaxPool1d::new(3, 2, 1);

        // Residual layers
        let layer1 = Self::make_layer(64, 64, 2, 1);
        let layer2 = Self::make_layer(64, 128, 2, 2);
        let layer3 = Self::make_layer(128, 256, 2, 2);
        let layer4 = Self::make_layer(256, 512, 2, 2);

        // Output layers
        let avgpool = AdaptiveAvgPool1d::new(1);
        let fc = Linear::new(512, num_classes, true);

        Self {
            conv1,
            bn1,
            relu,
            maxpool,
            layer1,
            layer2,
            layer3,
            layer4,
            avgpool,
            fc,
            in_channels,
            num_classes,
        }
    }

    /// Create a layer with multiple blocks
    fn make_layer(
        in_channels: usize,
        out_channels: usize,
        num_blocks: usize,
        stride: usize,
    ) -> Vec<BasicBlock> {
        let mut blocks = Vec::with_capacity(num_blocks);

        // First block may have stride and change dimensions
        blocks.push(BasicBlock::new(in_channels, out_channels, stride));

        // Remaining blocks maintain dimensions
        for _ in 1..num_blocks {
            blocks.push(BasicBlock::new(out_channels, out_channels, 1));
        }

        blocks
    }

    /// Forward pass
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [batch, channels, length]
    ///
    /// # Returns
    ///
    /// Output tensor of shape [batch, num_classes]
    pub fn forward(&self, x: &Array3<f32>) -> Array2<f32> {
        // Initial layers
        let mut out = self.conv1.forward(x);
        out = self.bn1.forward(&out);
        out = self.relu.forward(&out);
        out = self.maxpool.forward(&out);

        // Layer 1
        for block in &self.layer1 {
            out = block.forward(&out);
        }

        // Layer 2
        for block in &self.layer2 {
            out = block.forward(&out);
        }

        // Layer 3
        for block in &self.layer3 {
            out = block.forward(&out);
        }

        // Layer 4
        for block in &self.layer4 {
            out = block.forward(&out);
        }

        // Global average pooling and flatten
        out = self.avgpool.forward(&out);
        let batch_size = out.shape()[0];
        let channels = out.shape()[1];
        let flat = out.into_shape_with_order((batch_size, channels)).unwrap();

        // Fully connected layer
        self.fc.forward(&flat)
    }

    /// Predict class probabilities using softmax
    pub fn predict_proba(&self, x: &Array3<f32>) -> Array2<f32> {
        let logits = self.forward(x);
        softmax(&logits)
    }

    /// Predict class labels
    pub fn predict(&self, x: &Array3<f32>) -> Vec<usize> {
        let probs = self.predict_proba(x);
        let batch_size = probs.shape()[0];

        (0..batch_size)
            .map(|b| {
                let row = probs.row(b);
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Get total number of parameters
    pub fn num_params(&self) -> usize {
        let mut params = 0;

        params += self.conv1.num_params();
        params += self.bn1.num_features * 2;

        for block in &self.layer1 {
            params += block.num_params();
        }
        for block in &self.layer2 {
            params += block.num_params();
        }
        for block in &self.layer3 {
            params += block.num_params();
        }
        for block in &self.layer4 {
            params += block.num_params();
        }

        params += self.fc.num_params();

        params
    }
}

/// ResNet-34 model for time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResNet34 {
    /// Initial convolution
    pub conv1: Conv1d,
    /// Initial batch norm
    pub bn1: BatchNorm1d,
    /// ReLU activation
    pub relu: ReLU,
    /// Max pooling
    pub maxpool: MaxPool1d,
    /// Layer 1 (3 blocks)
    pub layer1: Vec<BasicBlock>,
    /// Layer 2 (4 blocks)
    pub layer2: Vec<BasicBlock>,
    /// Layer 3 (6 blocks)
    pub layer3: Vec<BasicBlock>,
    /// Layer 4 (3 blocks)
    pub layer4: Vec<BasicBlock>,
    /// Adaptive average pooling
    pub avgpool: AdaptiveAvgPool1d,
    /// Final fully connected layer
    pub fc: Linear,
    /// Number of input channels
    pub in_channels: usize,
    /// Number of output classes
    pub num_classes: usize,
}

impl ResNet34 {
    /// Create a new ResNet-34 model
    pub fn new(in_channels: usize, num_classes: usize) -> Self {
        let conv1 = Conv1d::new(in_channels, 64, 7, 2, 3, false);
        let bn1 = BatchNorm1d::new(64);
        let relu = ReLU::new();
        let maxpool = MaxPool1d::new(3, 2, 1);

        let layer1 = Self::make_layer(64, 64, 3, 1);
        let layer2 = Self::make_layer(64, 128, 4, 2);
        let layer3 = Self::make_layer(128, 256, 6, 2);
        let layer4 = Self::make_layer(256, 512, 3, 2);

        let avgpool = AdaptiveAvgPool1d::new(1);
        let fc = Linear::new(512, num_classes, true);

        Self {
            conv1,
            bn1,
            relu,
            maxpool,
            layer1,
            layer2,
            layer3,
            layer4,
            avgpool,
            fc,
            in_channels,
            num_classes,
        }
    }

    fn make_layer(
        in_channels: usize,
        out_channels: usize,
        num_blocks: usize,
        stride: usize,
    ) -> Vec<BasicBlock> {
        let mut blocks = Vec::with_capacity(num_blocks);
        blocks.push(BasicBlock::new(in_channels, out_channels, stride));
        for _ in 1..num_blocks {
            blocks.push(BasicBlock::new(out_channels, out_channels, 1));
        }
        blocks
    }

    /// Forward pass
    pub fn forward(&self, x: &Array3<f32>) -> Array2<f32> {
        let mut out = self.conv1.forward(x);
        out = self.bn1.forward(&out);
        out = self.relu.forward(&out);
        out = self.maxpool.forward(&out);

        for block in &self.layer1 {
            out = block.forward(&out);
        }
        for block in &self.layer2 {
            out = block.forward(&out);
        }
        for block in &self.layer3 {
            out = block.forward(&out);
        }
        for block in &self.layer4 {
            out = block.forward(&out);
        }

        out = self.avgpool.forward(&out);
        let batch_size = out.shape()[0];
        let channels = out.shape()[1];
        let flat = out.into_shape_with_order((batch_size, channels)).unwrap();

        self.fc.forward(&flat)
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, x: &Array3<f32>) -> Array2<f32> {
        let logits = self.forward(x);
        softmax(&logits)
    }
}

/// ResNet-50 model with bottleneck blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResNet50 {
    /// Initial convolution
    pub conv1: Conv1d,
    /// Initial batch norm
    pub bn1: BatchNorm1d,
    /// ReLU activation
    pub relu: ReLU,
    /// Max pooling
    pub maxpool: MaxPool1d,
    /// Layer 1 (3 blocks)
    pub layer1: Vec<BottleneckBlock>,
    /// Layer 2 (4 blocks)
    pub layer2: Vec<BottleneckBlock>,
    /// Layer 3 (6 blocks)
    pub layer3: Vec<BottleneckBlock>,
    /// Layer 4 (3 blocks)
    pub layer4: Vec<BottleneckBlock>,
    /// Adaptive average pooling
    pub avgpool: AdaptiveAvgPool1d,
    /// Final fully connected layer
    pub fc: Linear,
    /// Number of input channels
    pub in_channels: usize,
    /// Number of output classes
    pub num_classes: usize,
}

impl ResNet50 {
    /// Create a new ResNet-50 model
    pub fn new(in_channels: usize, num_classes: usize) -> Self {
        let conv1 = Conv1d::new(in_channels, 64, 7, 2, 3, false);
        let bn1 = BatchNorm1d::new(64);
        let relu = ReLU::new();
        let maxpool = MaxPool1d::new(3, 2, 1);

        let layer1 = Self::make_layer(64, 64, 3, 1);
        let layer2 = Self::make_layer(256, 128, 4, 2);
        let layer3 = Self::make_layer(512, 256, 6, 2);
        let layer4 = Self::make_layer(1024, 512, 3, 2);

        let avgpool = AdaptiveAvgPool1d::new(1);
        let fc = Linear::new(2048, num_classes, true); // 512 * 4 = 2048

        Self {
            conv1,
            bn1,
            relu,
            maxpool,
            layer1,
            layer2,
            layer3,
            layer4,
            avgpool,
            fc,
            in_channels,
            num_classes,
        }
    }

    fn make_layer(
        in_channels: usize,
        out_channels: usize,
        num_blocks: usize,
        stride: usize,
    ) -> Vec<BottleneckBlock> {
        let mut blocks = Vec::with_capacity(num_blocks);
        blocks.push(BottleneckBlock::new(in_channels, out_channels, stride));
        for _ in 1..num_blocks {
            blocks.push(BottleneckBlock::new(
                out_channels * BottleneckBlock::EXPANSION,
                out_channels,
                1,
            ));
        }
        blocks
    }

    /// Forward pass
    pub fn forward(&self, x: &Array3<f32>) -> Array2<f32> {
        let mut out = self.conv1.forward(x);
        out = self.bn1.forward(&out);
        out = self.relu.forward(&out);
        out = self.maxpool.forward(&out);

        for block in &self.layer1 {
            out = block.forward(&out);
        }
        for block in &self.layer2 {
            out = block.forward(&out);
        }
        for block in &self.layer3 {
            out = block.forward(&out);
        }
        for block in &self.layer4 {
            out = block.forward(&out);
        }

        out = self.avgpool.forward(&out);
        let batch_size = out.shape()[0];
        let channels = out.shape()[1];
        let flat = out.into_shape_with_order((batch_size, channels)).unwrap();

        self.fc.forward(&flat)
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, x: &Array3<f32>) -> Array2<f32> {
        let logits = self.forward(x);
        softmax(&logits)
    }
}

/// Softmax function for probability conversion
fn softmax(logits: &Array2<f32>) -> Array2<f32> {
    let mut probs = logits.clone();

    for mut row in probs.rows_mut() {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();

        for val in row.iter_mut() {
            *val = (*val - max).exp() / exp_sum;
        }
    }

    probs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resnet18_forward() {
        let model = ResNet18::new(5, 3);
        let input = Array3::ones((2, 5, 256));
        let output = model.forward(&input);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_resnet18_predict() {
        let model = ResNet18::new(5, 3);
        let input = Array3::ones((4, 5, 256));
        let predictions = model.predict(&input);
        assert_eq!(predictions.len(), 4);
        assert!(predictions.iter().all(|&p| p < 3));
    }

    #[test]
    fn test_resnet34_forward() {
        let model = ResNet34::new(5, 3);
        let input = Array3::ones((2, 5, 256));
        let output = model.forward(&input);
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_softmax() {
        let logits = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let probs = softmax(&logits);

        // Check that probabilities sum to 1
        let sum: f32 = probs.row(0).iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Check that max is at the right position
        let max_idx = probs
            .row(0)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        assert_eq!(max_idx, 2);
    }
}
