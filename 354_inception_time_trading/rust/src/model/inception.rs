//! Inception Module for Time Series
//!
//! This module implements the core Inception module that uses
//! multiple parallel convolutions with different kernel sizes
//! to capture patterns at multiple time scales.

use anyhow::Result;
use tch::{nn, Tensor};

/// Configuration for an Inception module
#[derive(Debug, Clone)]
pub struct InceptionConfig {
    /// Number of input channels
    pub in_channels: i64,
    /// Number of filters per convolution branch
    pub num_filters: i64,
    /// Kernel sizes for the convolution branches
    pub kernel_sizes: Vec<i64>,
    /// Bottleneck size (reduces dimensionality before convolutions)
    pub bottleneck_size: i64,
    /// Whether to use batch normalization
    pub use_batch_norm: bool,
}

impl Default for InceptionConfig {
    fn default() -> Self {
        Self {
            in_channels: 1,
            num_filters: 32,
            kernel_sizes: vec![10, 20, 40],
            bottleneck_size: 32,
            use_batch_norm: true,
        }
    }
}

/// Inception module for time series
///
/// The module consists of:
/// 1. A bottleneck layer (1x1 convolution) to reduce dimensionality
/// 2. Multiple parallel convolutions with different kernel sizes
/// 3. A max pooling branch with 1x1 convolution
/// 4. Concatenation of all branches
/// 5. Batch normalization and ReLU activation
#[derive(Debug)]
pub struct InceptionModule {
    /// Bottleneck convolution (1x1)
    bottleneck: nn::Conv1D,
    /// Convolution branches with different kernel sizes
    conv_branches: Vec<nn::Conv1D>,
    /// Max pooling layer
    max_pool: nn::MaxPool1D,
    /// Convolution after max pooling
    pool_conv: nn::Conv1D,
    /// Batch normalization layer
    batch_norm: Option<nn::BatchNorm>,
    /// Configuration
    config: InceptionConfig,
}

impl InceptionModule {
    /// Create a new Inception module
    pub fn new(vs: &nn::Path, config: InceptionConfig) -> Result<Self> {
        // Bottleneck layer
        let bottleneck = nn::conv1d(
            vs / "bottleneck",
            config.in_channels,
            config.bottleneck_size,
            1,
            nn::ConvConfig {
                padding: 0,
                bias: false,
                ..Default::default()
            },
        );

        // Convolution branches
        let mut conv_branches = Vec::new();
        for (i, &kernel_size) in config.kernel_sizes.iter().enumerate() {
            let padding = kernel_size / 2;
            let conv = nn::conv1d(
                vs / format!("conv_{}", i),
                config.bottleneck_size,
                config.num_filters,
                kernel_size,
                nn::ConvConfig {
                    padding,
                    bias: false,
                    ..Default::default()
                },
            );
            conv_branches.push(conv);
        }

        // Max pooling branch
        let max_pool = nn::MaxPool1D {
            kernel_size: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            ceil_mode: false,
        };

        let pool_conv = nn::conv1d(
            vs / "pool_conv",
            config.in_channels,
            config.num_filters,
            1,
            nn::ConvConfig {
                padding: 0,
                bias: false,
                ..Default::default()
            },
        );

        // Batch normalization
        let out_channels = config.num_filters * (config.kernel_sizes.len() as i64 + 1);
        let batch_norm = if config.use_batch_norm {
            Some(nn::batch_norm1d(vs / "bn", out_channels, Default::default()))
        } else {
            None
        };

        Ok(Self {
            bottleneck,
            conv_branches,
            max_pool,
            pool_conv,
            batch_norm,
            config,
        })
    }

    /// Forward pass through the Inception module
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape (batch, channels, sequence_length)
    ///
    /// # Returns
    /// Output tensor of shape (batch, out_channels, sequence_length)
    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        // Bottleneck
        let bottleneck_out = x.apply(&self.bottleneck);

        // Convolution branches
        let mut branch_outputs: Vec<Tensor> = self
            .conv_branches
            .iter()
            .map(|conv| bottleneck_out.apply(conv))
            .collect();

        // Max pooling branch
        let pool_out = self.max_pool.forward(x);
        let pool_conv_out = pool_out.apply(&self.pool_conv);
        branch_outputs.push(pool_conv_out);

        // Concatenate all branches along channel dimension
        let concat = Tensor::cat(&branch_outputs, 1);

        // Batch normalization and ReLU
        let normalized = if let Some(ref bn) = self.batch_norm {
            concat.apply_t(bn, train)
        } else {
            concat
        };

        normalized.relu()
    }

    /// Get the number of output channels
    pub fn out_channels(&self) -> i64 {
        self.config.num_filters * (self.config.kernel_sizes.len() as i64 + 1)
    }
}

/// Residual connection that can adjust dimensions if needed
#[derive(Debug)]
pub struct ResidualConnection {
    /// Optional 1x1 convolution for dimension matching
    conv: Option<nn::Conv1D>,
    /// Batch normalization for the shortcut
    batch_norm: Option<nn::BatchNorm>,
}

impl ResidualConnection {
    /// Create a new residual connection
    pub fn new(vs: &nn::Path, in_channels: i64, out_channels: i64, use_batch_norm: bool) -> Self {
        let (conv, batch_norm) = if in_channels != out_channels {
            let conv = nn::conv1d(
                vs / "residual_conv",
                in_channels,
                out_channels,
                1,
                nn::ConvConfig {
                    padding: 0,
                    bias: false,
                    ..Default::default()
                },
            );
            let bn = if use_batch_norm {
                Some(nn::batch_norm1d(
                    vs / "residual_bn",
                    out_channels,
                    Default::default(),
                ))
            } else {
                None
            };
            (Some(conv), bn)
        } else {
            (None, None)
        };

        Self { conv, batch_norm }
    }

    /// Apply residual connection
    pub fn forward(&self, input: &Tensor, output: &Tensor, train: bool) -> Tensor {
        let shortcut = if let Some(ref conv) = self.conv {
            let x = input.apply(conv);
            if let Some(ref bn) = self.batch_norm {
                x.apply_t(bn, train)
            } else {
                x
            }
        } else {
            input.shallow_clone()
        };

        (output + shortcut).relu()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inception_config_default() {
        let config = InceptionConfig::default();
        assert_eq!(config.num_filters, 32);
        assert_eq!(config.kernel_sizes, vec![10, 20, 40]);
    }
}
