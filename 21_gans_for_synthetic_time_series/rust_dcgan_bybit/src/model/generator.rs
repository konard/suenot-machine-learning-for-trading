//! Generator network for DCGAN
//!
//! The Generator transforms random noise vectors into synthetic time series data.
//! Architecture uses transposed 1D convolutions to upsample from latent space.

use tch::{nn, nn::Module, nn::ModuleT, Device, Tensor};

/// Generator network configuration
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Size of the latent noise vector
    pub latent_dim: i64,
    /// Length of output sequence
    pub sequence_length: i64,
    /// Number of output features (e.g., OHLCV = 5)
    pub num_features: i64,
    /// Base number of filters
    pub base_filters: i64,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            latent_dim: 100,
            sequence_length: 24,  // e.g., 24 hours
            num_features: 5,      // OHLCV
            base_filters: 256,
        }
    }
}

/// Generator network
///
/// Architecture:
/// 1. Dense layer from latent space to initial feature map
/// 2. Series of ConvTranspose1d layers with BatchNorm and LeakyReLU
/// 3. Final ConvTranspose1d with Tanh activation
#[derive(Debug)]
pub struct Generator {
    config: GeneratorConfig,
    /// Initial dense projection
    fc: nn::Linear,
    /// Transposed convolution layers
    conv1: nn::ConvTranspose1D,
    bn1: nn::BatchNorm,
    conv2: nn::ConvTranspose1D,
    bn2: nn::BatchNorm,
    conv3: nn::ConvTranspose1D,
    bn3: nn::BatchNorm,
    conv4: nn::ConvTranspose1D,
}

impl Generator {
    /// Create a new Generator network
    pub fn new(vs: &nn::Path, config: GeneratorConfig) -> Self {
        let base = config.base_filters;

        // Calculate initial projection size
        // We'll project to (base_filters, sequence_length/8) then upsample
        let init_seq_len = config.sequence_length / 8;
        let init_size = base * init_seq_len;

        let fc = nn::linear(vs / "fc", config.latent_dim, init_size, Default::default());

        // ConvTranspose1d: (in_channels, out_channels, kernel_size)
        let conv_config = nn::ConvTransposeConfig {
            stride: 2,
            padding: 1,
            output_padding: 1,
            ..Default::default()
        };

        let conv1 = nn::conv_transpose1d(vs / "conv1", base, base / 2, 4, conv_config);
        let bn1 = nn::batch_norm1d(vs / "bn1", base / 2, Default::default());

        let conv2 = nn::conv_transpose1d(vs / "conv2", base / 2, base / 4, 4, conv_config);
        let bn2 = nn::batch_norm1d(vs / "bn2", base / 4, Default::default());

        let conv3 = nn::conv_transpose1d(vs / "conv3", base / 4, base / 8, 4, conv_config);
        let bn3 = nn::batch_norm1d(vs / "bn3", base / 8, Default::default());

        // Final layer: no batch norm, tanh activation
        let conv4_config = nn::ConvTransposeConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        };
        let conv4 = nn::conv_transpose1d(
            vs / "conv4",
            base / 8,
            config.num_features,
            3,
            conv4_config,
        );

        Self {
            config,
            fc,
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            conv4,
        }
    }

    /// Generate synthetic samples from noise
    ///
    /// # Arguments
    ///
    /// * `noise` - Tensor of shape (batch_size, latent_dim)
    /// * `train` - Whether in training mode (affects batch norm)
    ///
    /// # Returns
    ///
    /// Tensor of shape (batch_size, sequence_length, num_features)
    pub fn forward_t(&self, noise: &Tensor, train: bool) -> Tensor {
        let batch_size = noise.size()[0];
        let base = self.config.base_filters;
        let init_seq_len = self.config.sequence_length / 8;

        // Project and reshape: (batch, latent) -> (batch, channels, seq_len)
        let x = self.fc.forward(noise);
        let x = x.view([batch_size, base, init_seq_len]);

        // Upsample through transposed convolutions
        let x = self.conv1.forward(&x);
        let x = self.bn1.forward_t(&x, train);
        let x = x.leaky_relu();

        let x = self.conv2.forward(&x);
        let x = self.bn2.forward_t(&x, train);
        let x = x.leaky_relu();

        let x = self.conv3.forward(&x);
        let x = self.bn3.forward_t(&x, train);
        let x = x.leaky_relu();

        let x = self.conv4.forward(&x);
        let x = x.tanh();

        // Transpose to (batch, seq_len, features) and ensure correct sequence length
        let x = x.transpose(1, 2);

        // Adjust sequence length if needed
        let current_len = x.size()[1];
        if current_len != self.config.sequence_length {
            x.narrow(1, 0, self.config.sequence_length)
        } else {
            x
        }
    }

    /// Generate samples (inference mode)
    pub fn generate(&self, noise: &Tensor) -> Tensor {
        self.forward_t(noise, false)
    }

    /// Generate random samples
    ///
    /// # Arguments
    ///
    /// * `num_samples` - Number of samples to generate
    /// * `device` - Device to create tensors on
    pub fn generate_random(&self, num_samples: i64, device: Device) -> Tensor {
        let noise = Tensor::randn([num_samples, self.config.latent_dim], (tch::Kind::Float, device));
        self.generate(&noise)
    }

    /// Get configuration
    pub fn config(&self) -> &GeneratorConfig {
        &self.config
    }
}

impl ModuleT for Generator {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        Generator::forward_t(self, xs, train)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::nn::VarStore;

    #[test]
    fn test_generator_output_shape() {
        let vs = VarStore::new(Device::Cpu);
        let config = GeneratorConfig {
            latent_dim: 100,
            sequence_length: 24,
            num_features: 5,
            base_filters: 256,
        };
        let gen = Generator::new(&vs.root(), config);

        let noise = Tensor::randn([4, 100], (tch::Kind::Float, Device::Cpu));
        let output = gen.generate(&noise);

        assert_eq!(output.size(), vec![4, 24, 5]);
    }
}
