//! Dense Block Implementation
//!
//! A dense block contains multiple dense layers, where each layer
//! is connected to ALL previous layers (feature concatenation).

use ndarray::{Array2, Axis, concatenate};
use super::layer::DenseLayer;

/// Dense Block: each layer connected to all previous layers
///
/// Layer 0: x_0 → H_0(x_0) = y_0
/// Layer 1: [x_0, y_0] → H_1([x_0, y_0]) = y_1
/// Layer 2: [x_0, y_0, y_1] → H_2([x_0, y_0, y_1]) = y_2
/// ...
#[derive(Debug, Clone)]
pub struct DenseBlock {
    /// Number of layers in this block
    pub num_layers: usize,

    /// Input channels to the block
    pub in_channels: usize,

    /// Growth rate (new channels per layer)
    pub growth_rate: usize,

    /// Output channels (in_channels + num_layers * growth_rate)
    pub out_channels: usize,

    /// The dense layers
    pub layers: Vec<DenseLayer>,
}

impl DenseBlock {
    /// Create a new dense block
    pub fn new(
        num_layers: usize,
        in_channels: usize,
        growth_rate: usize,
        dropout: f64,
        use_bottleneck: bool,
        bottleneck_factor: usize,
    ) -> Self {
        let mut layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            // Each layer receives all previous feature maps
            let layer_in_channels = in_channels + i * growth_rate;

            let layer = DenseLayer::new(
                layer_in_channels,
                growth_rate,
                3, // kernel size
                dropout,
                use_bottleneck,
                bottleneck_factor,
            );

            layers.push(layer);
        }

        let out_channels = in_channels + num_layers * growth_rate;

        Self {
            num_layers,
            in_channels,
            growth_rate,
            out_channels,
            layers,
        }
    }

    /// Forward pass through the dense block
    ///
    /// Input: [in_channels, sequence_length]
    /// Output: [out_channels, sequence_length] where out_channels = in_channels + num_layers * growth_rate
    pub fn forward(&self, x: &Array2<f64>, training: bool) -> Array2<f64> {
        let mut features: Vec<Array2<f64>> = vec![x.clone()];

        for layer in &self.layers {
            // Concatenate ALL previous feature maps along channel axis
            let views: Vec<_> = features.iter().map(|f| f.view()).collect();
            let concat_features = concatenate(Axis(0), &views).unwrap();

            // Forward through current layer
            let new_features = layer.forward(&concat_features, training);

            // Add new features to the list
            features.push(new_features);
        }

        // Return concatenation of all features
        let views: Vec<_> = features.iter().map(|f| f.view()).collect();
        concatenate(Axis(0), &views).unwrap()
    }

    /// Get total number of parameters in this block
    pub fn num_parameters(&self) -> usize {
        self.layers.iter().map(|l| l.num_parameters()).sum()
    }

    /// Get information about the block
    pub fn info(&self) -> DenseBlockInfo {
        DenseBlockInfo {
            num_layers: self.num_layers,
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            growth_rate: self.growth_rate,
            parameters: self.num_parameters(),
        }
    }
}

/// Information about a dense block
#[derive(Debug, Clone)]
pub struct DenseBlockInfo {
    pub num_layers: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub growth_rate: usize,
    pub parameters: usize,
}

impl std::fmt::Display for DenseBlockInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DenseBlock(layers={}, in={}, out={}, k={}, params={})",
            self.num_layers,
            self.in_channels,
            self.out_channels,
            self.growth_rate,
            self.parameters
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_dense_block_creation() {
        let block = DenseBlock::new(6, 64, 32, 0.2, true, 4);
        assert_eq!(block.num_layers, 6);
        assert_eq!(block.in_channels, 64);
        assert_eq!(block.out_channels, 64 + 6 * 32); // 256
    }

    #[test]
    fn test_dense_block_forward() {
        let block = DenseBlock::new(4, 32, 16, 0.0, false, 4);
        let input = Array2::ones((32, 64));
        let output = block.forward(&input, false);

        // Output channels = 32 + 4 * 16 = 96
        assert_eq!(output.dim().0, 96);
        assert_eq!(output.dim().1, 64); // Sequence length preserved
    }

    #[test]
    fn test_dense_block_info() {
        let block = DenseBlock::new(6, 64, 32, 0.2, true, 4);
        let info = block.info();
        assert_eq!(info.num_layers, 6);
        assert_eq!(info.out_channels, 256);
    }

    #[test]
    fn test_channel_growth() {
        // Verify that channels grow correctly
        let growth_rate = 24;
        let num_layers = 8;
        let in_channels = 48;

        let block = DenseBlock::new(num_layers, in_channels, growth_rate, 0.0, false, 4);

        let expected_out = in_channels + num_layers * growth_rate;
        assert_eq!(block.out_channels, expected_out);
    }
}
