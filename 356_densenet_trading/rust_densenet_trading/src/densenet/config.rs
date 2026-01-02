//! DenseNet Configuration
//!
//! Configuration parameters for building DenseNet architectures.

use serde::{Deserialize, Serialize};

/// Configuration for DenseNet architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseNetConfig {
    /// Number of input features (OHLCV + indicators)
    pub input_features: usize,

    /// Sequence length (lookback window)
    pub sequence_length: usize,

    /// Growth rate (k): number of new features each layer produces
    pub growth_rate: usize,

    /// Number of layers in each dense block
    pub block_config: Vec<usize>,

    /// Compression factor for transition layers (0 < θ <= 1)
    pub compression: f64,

    /// Number of output classes (Long, Hold, Short)
    pub num_classes: usize,

    /// Dropout rate for regularization
    pub dropout: f64,

    /// Initial number of channels after first convolution
    pub init_channels: usize,

    /// Whether to use bottleneck layers (1x1 conv before 3x1)
    pub use_bottleneck: bool,

    /// Bottleneck factor (multiply growth_rate for intermediate channels)
    pub bottleneck_factor: usize,
}

impl Default for DenseNetConfig {
    fn default() -> Self {
        Self {
            input_features: 32,
            sequence_length: 128,
            growth_rate: 32,
            block_config: vec![6, 12, 24],
            compression: 0.5,
            num_classes: 3,
            dropout: 0.2,
            init_channels: 64,
            use_bottleneck: true,
            bottleneck_factor: 4,
        }
    }
}

impl DenseNetConfig {
    /// Create a tiny DenseNet for real-time HFT
    pub fn tiny() -> Self {
        Self {
            input_features: 16,
            sequence_length: 64,
            growth_rate: 16,
            block_config: vec![4, 4, 4],
            compression: 0.5,
            num_classes: 3,
            dropout: 0.1,
            init_channels: 32,
            use_bottleneck: false,
            bottleneck_factor: 4,
        }
    }

    /// Create a small DenseNet for intraday trading
    pub fn small() -> Self {
        Self {
            input_features: 24,
            sequence_length: 96,
            growth_rate: 24,
            block_config: vec![6, 12, 8],
            compression: 0.5,
            num_classes: 3,
            dropout: 0.15,
            init_channels: 48,
            use_bottleneck: true,
            bottleneck_factor: 4,
        }
    }

    /// Create a medium DenseNet for swing trading
    pub fn medium() -> Self {
        Self::default()
    }

    /// Create a large DenseNet for research
    pub fn large() -> Self {
        Self {
            input_features: 48,
            sequence_length: 256,
            growth_rate: 48,
            block_config: vec![6, 12, 32, 24],
            compression: 0.5,
            num_classes: 3,
            dropout: 0.3,
            init_channels: 96,
            use_bottleneck: true,
            bottleneck_factor: 4,
        }
    }

    /// Calculate total number of parameters (approximate)
    pub fn estimate_parameters(&self) -> usize {
        let mut total = 0;
        let mut channels = self.init_channels;

        // Initial convolution
        total += self.input_features * channels * 7;

        // Dense blocks and transitions
        for (i, &num_layers) in self.block_config.iter().enumerate() {
            // Dense block
            for j in 0..num_layers {
                let in_ch = channels + j * self.growth_rate;
                if self.use_bottleneck {
                    // Bottleneck: 1x1 conv
                    total += in_ch * self.bottleneck_factor * self.growth_rate;
                    // 3x1 conv
                    total += self.bottleneck_factor * self.growth_rate * self.growth_rate * 3;
                } else {
                    // 3x1 conv only
                    total += in_ch * self.growth_rate * 3;
                }
            }
            channels += num_layers * self.growth_rate;

            // Transition layer (except last block)
            if i < self.block_config.len() - 1 {
                let out_ch = (channels as f64 * self.compression) as usize;
                total += channels * out_ch;
                channels = out_ch;
            }
        }

        // Classifier
        total += channels * 128 + 128 * self.num_classes;

        total
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.growth_rate == 0 {
            return Err("Growth rate must be positive".to_string());
        }
        if self.compression <= 0.0 || self.compression > 1.0 {
            return Err("Compression must be in (0, 1]".to_string());
        }
        if self.block_config.is_empty() {
            return Err("Block config must not be empty".to_string());
        }
        if self.dropout < 0.0 || self.dropout >= 1.0 {
            return Err("Dropout must be in [0, 1)".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DenseNetConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_all_presets() {
        for config in [
            DenseNetConfig::tiny(),
            DenseNetConfig::small(),
            DenseNetConfig::medium(),
            DenseNetConfig::large(),
        ] {
            assert!(config.validate().is_ok());
        }
    }

    #[test]
    fn test_parameter_estimation() {
        let config = DenseNetConfig::tiny();
        let params = config.estimate_parameters();
        assert!(params > 0);
        assert!(params < 1_000_000); // Tiny should be under 1M params
    }
}
