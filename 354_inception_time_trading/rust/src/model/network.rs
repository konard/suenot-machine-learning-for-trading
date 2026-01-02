//! Full InceptionTime Network
//!
//! This module implements the complete InceptionTime architecture
//! with multiple Inception modules and residual connections.

use anyhow::Result;
use tch::{nn, Device, Tensor};

use super::inception::{InceptionConfig, InceptionModule, ResidualConnection};

/// Configuration for the full InceptionTime network
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Number of input features
    pub input_features: i64,
    /// Number of output classes
    pub num_classes: i64,
    /// Number of filters in each Inception module
    pub num_filters: i64,
    /// Number of Inception modules (depth)
    pub depth: i64,
    /// Kernel sizes for convolutions
    pub kernel_sizes: Vec<i64>,
    /// Bottleneck size
    pub bottleneck_size: i64,
    /// Add residual connection every N modules
    pub residual_interval: i64,
    /// Dropout rate
    pub dropout: f64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            input_features: 15,
            num_classes: 3,
            num_filters: 32,
            depth: 6,
            kernel_sizes: vec![10, 20, 40],
            bottleneck_size: 32,
            residual_interval: 3,
            dropout: 0.2,
        }
    }
}

/// InceptionTime network for time series classification
#[derive(Debug)]
pub struct InceptionTimeNetwork {
    /// Inception modules
    inception_modules: Vec<InceptionModule>,
    /// Residual connections
    residual_connections: Vec<Option<ResidualConnection>>,
    /// Global average pooling (handled in forward)
    /// Final classification layer
    classifier: nn::Linear,
    /// Dropout layer
    dropout: f64,
    /// Configuration
    config: NetworkConfig,
    /// Device
    device: Device,
}

impl InceptionTimeNetwork {
    /// Create a new InceptionTime network
    pub fn new(vs: &nn::Path, config: NetworkConfig) -> Result<Self> {
        let mut inception_modules = Vec::new();
        let mut residual_connections = Vec::new();

        let mut current_channels = config.input_features;

        for i in 0..config.depth {
            // Create Inception module
            let inception_config = InceptionConfig {
                in_channels: current_channels,
                num_filters: config.num_filters,
                kernel_sizes: config.kernel_sizes.clone(),
                bottleneck_size: config.bottleneck_size,
                use_batch_norm: true,
            };

            let inception = InceptionModule::new(&(vs / format!("inception_{}", i)), inception_config)?;
            let out_channels = inception.out_channels();

            // Create residual connection if needed
            let residual = if (i + 1) % config.residual_interval as i64 == 0 {
                Some(ResidualConnection::new(
                    &(vs / format!("residual_{}", i)),
                    current_channels,
                    out_channels,
                    true,
                ))
            } else {
                None
            };

            inception_modules.push(inception);
            residual_connections.push(residual);

            current_channels = out_channels;
        }

        // Final classifier
        let classifier = nn::linear(
            vs / "classifier",
            current_channels,
            config.num_classes,
            Default::default(),
        );

        let device = vs.device();

        Ok(Self {
            inception_modules,
            residual_connections,
            classifier,
            dropout: config.dropout,
            config,
            device,
        })
    }

    /// Forward pass through the network
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape (batch, sequence_length, features)
    /// * `train` - Whether in training mode (affects dropout and batch norm)
    ///
    /// # Returns
    /// Output tensor of shape (batch, num_classes)
    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        // Transpose to (batch, features, sequence_length) for conv1d
        let mut out = x.permute([0, 2, 1]);
        let mut residual_input = out.shallow_clone();

        for (i, (inception, residual)) in self
            .inception_modules
            .iter()
            .zip(self.residual_connections.iter())
            .enumerate()
        {
            out = inception.forward(&out, train);

            // Apply residual connection if present
            if let Some(ref res) = residual {
                out = res.forward(&residual_input, &out, train);
                residual_input = out.shallow_clone();
            }

            // Check for residual interval to update residual_input
            if (i + 1) as i64 % self.config.residual_interval == 0 {
                residual_input = out.shallow_clone();
            }
        }

        // Global average pooling over time dimension
        let pooled = out.mean_dim(Some([2].as_slice()), false, tch::Kind::Float);

        // Apply dropout
        let dropped = if train && self.dropout > 0.0 {
            pooled.dropout(self.dropout, train)
        } else {
            pooled
        };

        // Classification
        dropped.apply(&self.classifier)
    }

    /// Get prediction probabilities
    pub fn predict_proba(&self, x: &Tensor) -> Tensor {
        let logits = self.forward(x, false);
        logits.softmax(-1, tch::Kind::Float)
    }

    /// Get predicted class
    pub fn predict(&self, x: &Tensor) -> Tensor {
        let probs = self.predict_proba(x);
        probs.argmax(-1, false)
    }

    /// Get device
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get configuration
    pub fn config(&self) -> &NetworkConfig {
        &self.config
    }

    /// Save model weights
    pub fn save(&self, _path: &str) -> Result<()> {
        // Note: Actual saving would use vs.save() at the VarStore level
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_config_default() {
        let config = NetworkConfig::default();
        assert_eq!(config.depth, 6);
        assert_eq!(config.num_classes, 3);
    }
}
