//! EfficientNet model configuration

use crate::EfficientNetVariant;

/// EfficientNet model configuration
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// EfficientNet variant
    pub variant: EfficientNetVariant,
    /// Input image size
    pub image_size: usize,
    /// Number of input channels (typically 3 for RGB)
    pub num_channels: usize,
    /// Number of output classes
    pub num_classes: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Whether to predict magnitude in addition to direction
    pub predict_magnitude: bool,
}

impl ModelConfig {
    /// Create a new model configuration
    pub fn new(variant: EfficientNetVariant) -> Self {
        Self {
            variant,
            image_size: variant.image_size(),
            num_channels: 3,
            num_classes: 3,
            dropout: 0.3,
            predict_magnitude: true,
        }
    }

    /// Create configuration for B0 variant
    pub fn b0() -> Self {
        Self::new(EfficientNetVariant::B0)
    }

    /// Create configuration for B2 variant (default)
    pub fn b2() -> Self {
        Self::new(EfficientNetVariant::B2)
    }

    /// Create configuration for B4 variant
    pub fn b4() -> Self {
        Self::new(EfficientNetVariant::B4)
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::b2()
    }
}

/// MBConv block configuration
#[derive(Debug, Clone, Copy)]
pub struct MBConvConfig {
    /// Input channels
    pub in_channels: usize,
    /// Output channels
    pub out_channels: usize,
    /// Kernel size (3 or 5)
    pub kernel_size: usize,
    /// Stride (1 or 2)
    pub stride: usize,
    /// Expansion ratio
    pub expansion_ratio: usize,
    /// Number of block repetitions
    pub num_repeat: usize,
}

/// Get MBConv block configurations for each EfficientNet variant
pub fn get_mbconv_configs(variant: EfficientNetVariant) -> Vec<MBConvConfig> {
    // Base configuration (B0)
    let base_configs = vec![
        MBConvConfig {
            in_channels: 32,
            out_channels: 16,
            kernel_size: 3,
            stride: 1,
            expansion_ratio: 1,
            num_repeat: 1,
        },
        MBConvConfig {
            in_channels: 16,
            out_channels: 24,
            kernel_size: 3,
            stride: 2,
            expansion_ratio: 6,
            num_repeat: 2,
        },
        MBConvConfig {
            in_channels: 24,
            out_channels: 40,
            kernel_size: 5,
            stride: 2,
            expansion_ratio: 6,
            num_repeat: 2,
        },
        MBConvConfig {
            in_channels: 40,
            out_channels: 80,
            kernel_size: 3,
            stride: 2,
            expansion_ratio: 6,
            num_repeat: 3,
        },
        MBConvConfig {
            in_channels: 80,
            out_channels: 112,
            kernel_size: 5,
            stride: 1,
            expansion_ratio: 6,
            num_repeat: 3,
        },
        MBConvConfig {
            in_channels: 112,
            out_channels: 192,
            kernel_size: 5,
            stride: 2,
            expansion_ratio: 6,
            num_repeat: 4,
        },
        MBConvConfig {
            in_channels: 192,
            out_channels: 320,
            kernel_size: 3,
            stride: 1,
            expansion_ratio: 6,
            num_repeat: 1,
        },
    ];

    // Scale based on variant
    let (width_mult, depth_mult) = match variant {
        EfficientNetVariant::B0 => (1.0, 1.0),
        EfficientNetVariant::B1 => (1.0, 1.1),
        EfficientNetVariant::B2 => (1.1, 1.2),
        EfficientNetVariant::B3 => (1.2, 1.4),
        EfficientNetVariant::B4 => (1.4, 1.8),
        EfficientNetVariant::B5 => (1.6, 2.2),
        EfficientNetVariant::B6 => (1.8, 2.6),
        EfficientNetVariant::B7 => (2.0, 3.1),
    };

    base_configs
        .into_iter()
        .map(|config| MBConvConfig {
            in_channels: scale_channels(config.in_channels, width_mult),
            out_channels: scale_channels(config.out_channels, width_mult),
            kernel_size: config.kernel_size,
            stride: config.stride,
            expansion_ratio: config.expansion_ratio,
            num_repeat: scale_depth(config.num_repeat, depth_mult),
        })
        .collect()
}

/// Scale channel count
fn scale_channels(channels: usize, multiplier: f64) -> usize {
    let scaled = (channels as f64 * multiplier).round() as usize;
    // Round to nearest multiple of 8
    ((scaled + 4) / 8) * 8
}

/// Scale depth (number of repeats)
fn scale_depth(depth: usize, multiplier: f64) -> usize {
    (depth as f64 * multiplier).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ModelConfig::default();
        assert_eq!(config.variant, EfficientNetVariant::B2);
        assert_eq!(config.image_size, 260);
        assert_eq!(config.num_classes, 3);
    }

    #[test]
    fn test_mbconv_configs() {
        let configs = get_mbconv_configs(EfficientNetVariant::B0);
        assert_eq!(configs.len(), 7);

        // B2 should have larger channels
        let configs_b2 = get_mbconv_configs(EfficientNetVariant::B2);
        assert!(configs_b2[0].out_channels >= configs[0].out_channels);
    }
}
