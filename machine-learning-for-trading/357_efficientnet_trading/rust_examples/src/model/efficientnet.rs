//! EfficientNet architecture configuration

/// EfficientNet variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfficientNetVariant {
    B0,
    B1,
    B2,
    B3,
    B4,
    B5,
    B6,
    B7,
}

impl EfficientNetVariant {
    /// Get input image size for this variant
    pub fn input_size(&self) -> u32 {
        match self {
            Self::B0 => 224,
            Self::B1 => 240,
            Self::B2 => 260,
            Self::B3 => 300,
            Self::B4 => 380,
            Self::B5 => 456,
            Self::B6 => 528,
            Self::B7 => 600,
        }
    }

    /// Get width multiplier
    pub fn width_coefficient(&self) -> f64 {
        match self {
            Self::B0 => 1.0,
            Self::B1 => 1.0,
            Self::B2 => 1.1,
            Self::B3 => 1.2,
            Self::B4 => 1.4,
            Self::B5 => 1.6,
            Self::B6 => 1.8,
            Self::B7 => 2.0,
        }
    }

    /// Get depth multiplier
    pub fn depth_coefficient(&self) -> f64 {
        match self {
            Self::B0 => 1.0,
            Self::B1 => 1.1,
            Self::B2 => 1.2,
            Self::B3 => 1.4,
            Self::B4 => 1.8,
            Self::B5 => 2.2,
            Self::B6 => 2.6,
            Self::B7 => 3.1,
        }
    }

    /// Get dropout rate
    pub fn dropout_rate(&self) -> f64 {
        match self {
            Self::B0 => 0.2,
            Self::B1 => 0.2,
            Self::B2 => 0.3,
            Self::B3 => 0.3,
            Self::B4 => 0.4,
            Self::B5 => 0.4,
            Self::B6 => 0.5,
            Self::B7 => 0.5,
        }
    }

    /// Get approximate parameter count in millions
    pub fn params_millions(&self) -> f64 {
        match self {
            Self::B0 => 5.3,
            Self::B1 => 7.8,
            Self::B2 => 9.2,
            Self::B3 => 12.0,
            Self::B4 => 19.0,
            Self::B5 => 30.0,
            Self::B6 => 43.0,
            Self::B7 => 66.0,
        }
    }
}

/// EfficientNet configuration
#[derive(Debug, Clone)]
pub struct EfficientNetConfig {
    pub variant: EfficientNetVariant,
    pub num_classes: usize,
    pub include_top: bool,
    pub dropout_rate: f64,
}

impl EfficientNetConfig {
    /// Create config for trading (3 classes: buy, hold, sell)
    pub fn for_trading(variant: EfficientNetVariant) -> Self {
        Self {
            variant,
            num_classes: 3,
            include_top: true,
            dropout_rate: variant.dropout_rate(),
        }
    }

    /// Create config for regression (1 output)
    pub fn for_regression(variant: EfficientNetVariant) -> Self {
        Self {
            variant,
            num_classes: 1,
            include_top: true,
            dropout_rate: variant.dropout_rate(),
        }
    }

    /// Create config for feature extraction (no classification head)
    pub fn for_features(variant: EfficientNetVariant) -> Self {
        Self {
            variant,
            num_classes: 0,
            include_top: false,
            dropout_rate: 0.0,
        }
    }

    /// Get input image size
    pub fn input_size(&self) -> u32 {
        self.variant.input_size()
    }
}

impl Default for EfficientNetConfig {
    fn default() -> Self {
        Self::for_trading(EfficientNetVariant::B0)
    }
}

/// Block configuration for EfficientNet stages
#[derive(Debug, Clone)]
pub struct StageConfig {
    pub kernel_size: usize,
    pub num_repeat: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub expand_ratio: usize,
    pub stride: usize,
    pub se_ratio: f64,
}

impl EfficientNetConfig {
    /// Get stage configurations for building the network
    pub fn stages(&self) -> Vec<StageConfig> {
        let width_mult = self.variant.width_coefficient();
        let depth_mult = self.variant.depth_coefficient();

        let scale_channels = |c: usize| -> usize {
            let c = (c as f64 * width_mult).ceil() as usize;
            // Round to nearest multiple of 8
            ((c + 4) / 8) * 8
        };

        let scale_depth = |d: usize| -> usize {
            (d as f64 * depth_mult).ceil() as usize
        };

        vec![
            StageConfig {
                kernel_size: 3,
                num_repeat: scale_depth(1),
                in_channels: scale_channels(32),
                out_channels: scale_channels(16),
                expand_ratio: 1,
                stride: 1,
                se_ratio: 0.25,
            },
            StageConfig {
                kernel_size: 3,
                num_repeat: scale_depth(2),
                in_channels: scale_channels(16),
                out_channels: scale_channels(24),
                expand_ratio: 6,
                stride: 2,
                se_ratio: 0.25,
            },
            StageConfig {
                kernel_size: 5,
                num_repeat: scale_depth(2),
                in_channels: scale_channels(24),
                out_channels: scale_channels(40),
                expand_ratio: 6,
                stride: 2,
                se_ratio: 0.25,
            },
            StageConfig {
                kernel_size: 3,
                num_repeat: scale_depth(3),
                in_channels: scale_channels(40),
                out_channels: scale_channels(80),
                expand_ratio: 6,
                stride: 2,
                se_ratio: 0.25,
            },
            StageConfig {
                kernel_size: 5,
                num_repeat: scale_depth(3),
                in_channels: scale_channels(80),
                out_channels: scale_channels(112),
                expand_ratio: 6,
                stride: 1,
                se_ratio: 0.25,
            },
            StageConfig {
                kernel_size: 5,
                num_repeat: scale_depth(4),
                in_channels: scale_channels(112),
                out_channels: scale_channels(192),
                expand_ratio: 6,
                stride: 2,
                se_ratio: 0.25,
            },
            StageConfig {
                kernel_size: 3,
                num_repeat: scale_depth(1),
                in_channels: scale_channels(192),
                out_channels: scale_channels(320),
                expand_ratio: 6,
                stride: 1,
                se_ratio: 0.25,
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variant_sizes() {
        assert_eq!(EfficientNetVariant::B0.input_size(), 224);
        assert_eq!(EfficientNetVariant::B3.input_size(), 300);
        assert_eq!(EfficientNetVariant::B7.input_size(), 600);
    }

    #[test]
    fn test_config_for_trading() {
        let config = EfficientNetConfig::for_trading(EfficientNetVariant::B0);
        assert_eq!(config.num_classes, 3);
        assert!(config.include_top);
    }

    #[test]
    fn test_stages() {
        let config = EfficientNetConfig::for_trading(EfficientNetVariant::B0);
        let stages = config.stages();
        assert_eq!(stages.len(), 7);
    }
}
