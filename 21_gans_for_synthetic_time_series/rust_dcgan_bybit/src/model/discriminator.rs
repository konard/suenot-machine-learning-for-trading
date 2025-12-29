//! Discriminator network for DCGAN
//!
//! The Discriminator classifies time series as real or fake.
//! Architecture uses 1D convolutions to downsample and extract features.

use tch::{nn, nn::Module, nn::ModuleT, Tensor};

/// Discriminator network configuration
#[derive(Debug, Clone)]
pub struct DiscriminatorConfig {
    /// Length of input sequence
    pub sequence_length: i64,
    /// Number of input features (e.g., OHLCV = 5)
    pub num_features: i64,
    /// Base number of filters
    pub base_filters: i64,
    /// Dropout rate
    pub dropout: f64,
}

impl Default for DiscriminatorConfig {
    fn default() -> Self {
        Self {
            sequence_length: 24,
            num_features: 5,
            base_filters: 64,
            dropout: 0.3,
        }
    }
}

/// Discriminator network
///
/// Architecture:
/// 1. Series of Conv1d layers with LeakyReLU and Dropout
/// 2. Flatten and Dense layer for final classification
#[derive(Debug)]
pub struct Discriminator {
    config: DiscriminatorConfig,
    /// Convolution layers
    conv1: nn::Conv1D,
    conv2: nn::Conv1D,
    conv3: nn::Conv1D,
    conv4: nn::Conv1D,
    /// Final classification layer
    fc: nn::Linear,
}

impl Discriminator {
    /// Create a new Discriminator network
    pub fn new(vs: &nn::Path, config: DiscriminatorConfig) -> Self {
        let base = config.base_filters;

        let conv_config = nn::ConvConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };

        // Input: (batch, features, seq_len) -> we'll transpose input
        let conv1 = nn::conv1d(vs / "conv1", config.num_features, base, 4, conv_config);
        let conv2 = nn::conv1d(vs / "conv2", base, base * 2, 4, conv_config);
        let conv3 = nn::conv1d(vs / "conv3", base * 2, base * 4, 4, conv_config);
        let conv4 = nn::conv1d(vs / "conv4", base * 4, base * 8, 4, conv_config);

        // Calculate flattened size after convolutions
        // Each conv with stride 2 roughly halves the sequence length
        let final_seq_len = config.sequence_length / 16;
        let flat_size = base * 8 * final_seq_len.max(1);

        let fc = nn::linear(vs / "fc", flat_size, 1, Default::default());

        Self {
            config,
            conv1,
            conv2,
            conv3,
            conv4,
            fc,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    ///
    /// * `input` - Tensor of shape (batch_size, sequence_length, num_features)
    /// * `train` - Whether in training mode (affects dropout)
    ///
    /// # Returns
    ///
    /// Tensor of shape (batch_size, 1) with logits (not sigmoid)
    pub fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        // Transpose to (batch, features, seq_len) for Conv1D
        let x = input.transpose(1, 2);

        // Apply convolutions with LeakyReLU and dropout
        let x = self.conv1.forward(&x);
        let x = x.leaky_relu();
        let x = x.dropout(self.config.dropout, train);

        let x = self.conv2.forward(&x);
        let x = x.leaky_relu();
        let x = x.dropout(self.config.dropout, train);

        let x = self.conv3.forward(&x);
        let x = x.leaky_relu();
        let x = x.dropout(self.config.dropout, train);

        let x = self.conv4.forward(&x);
        let x = x.leaky_relu();
        let x = x.dropout(self.config.dropout, train);

        // Flatten and classify
        let batch_size = x.size()[0];
        let x = x.view([batch_size, -1]);

        self.fc.forward(&x)
    }

    /// Classify samples (inference mode)
    ///
    /// Returns probability of being real (after sigmoid)
    pub fn classify(&self, input: &Tensor) -> Tensor {
        self.forward_t(input, false).sigmoid()
    }

    /// Get configuration
    pub fn config(&self) -> &DiscriminatorConfig {
        &self.config
    }
}

impl ModuleT for Discriminator {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        Discriminator::forward_t(self, xs, train)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn::VarStore, Device};

    #[test]
    fn test_discriminator_output_shape() {
        let vs = VarStore::new(Device::Cpu);
        let config = DiscriminatorConfig {
            sequence_length: 24,
            num_features: 5,
            base_filters: 64,
            dropout: 0.3,
        };
        let disc = Discriminator::new(&vs.root(), config);

        let input = Tensor::randn([4, 24, 5], (tch::Kind::Float, Device::Cpu));
        let output = disc.forward_t(&input, false);

        assert_eq!(output.size(), vec![4, 1]);
    }

    #[test]
    fn test_discriminator_classify() {
        let vs = VarStore::new(Device::Cpu);
        let config = DiscriminatorConfig::default();
        let disc = Discriminator::new(&vs.root(), config);

        let input = Tensor::randn([2, 24, 5], (tch::Kind::Float, Device::Cpu));
        let probs = disc.classify(&input);

        // Probabilities should be in [0, 1]
        let min_val: f64 = probs.min().double_value(&[]);
        let max_val: f64 = probs.max().double_value(&[]);
        assert!(min_val >= 0.0 && max_val <= 1.0);
    }
}
