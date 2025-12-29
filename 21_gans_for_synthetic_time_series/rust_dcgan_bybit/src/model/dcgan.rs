//! DCGAN wrapper combining Generator and Discriminator
//!
//! Provides convenient methods for training and generation.

use tch::{nn, nn::VarStore, Device, Tensor};

use super::discriminator::{Discriminator, DiscriminatorConfig};
use super::generator::{Generator, GeneratorConfig};

/// Complete DCGAN model
pub struct DCGAN {
    /// Generator network
    pub generator: Generator,
    /// Discriminator network
    pub discriminator: Discriminator,
    /// Variable store for generator
    pub gen_vs: VarStore,
    /// Variable store for discriminator
    pub disc_vs: VarStore,
    /// Device (CPU/GPU)
    pub device: Device,
}

impl DCGAN {
    /// Create a new DCGAN model
    ///
    /// # Arguments
    ///
    /// * `gen_config` - Generator configuration
    /// * `disc_config` - Discriminator configuration
    /// * `device` - Device to create model on
    pub fn new(gen_config: GeneratorConfig, disc_config: DiscriminatorConfig, device: Device) -> Self {
        let mut gen_vs = VarStore::new(device);
        let mut disc_vs = VarStore::new(device);

        let generator = Generator::new(&gen_vs.root(), gen_config);
        let discriminator = Discriminator::new(&disc_vs.root(), disc_config);

        Self {
            generator,
            discriminator,
            gen_vs,
            disc_vs,
            device,
        }
    }

    /// Create DCGAN with default configuration for given sequence parameters
    ///
    /// # Arguments
    ///
    /// * `sequence_length` - Length of time series sequences
    /// * `num_features` - Number of features (e.g., 5 for OHLCV)
    /// * `latent_dim` - Size of latent noise vector
    /// * `device` - Device to create model on
    pub fn with_defaults(
        sequence_length: i64,
        num_features: i64,
        latent_dim: i64,
        device: Device,
    ) -> Self {
        let gen_config = GeneratorConfig {
            latent_dim,
            sequence_length,
            num_features,
            base_filters: 256,
        };

        let disc_config = DiscriminatorConfig {
            sequence_length,
            num_features,
            base_filters: 64,
            dropout: 0.3,
        };

        Self::new(gen_config, disc_config, device)
    }

    /// Generate synthetic samples
    ///
    /// # Arguments
    ///
    /// * `num_samples` - Number of samples to generate
    ///
    /// # Returns
    ///
    /// Tensor of shape (num_samples, sequence_length, num_features)
    pub fn generate(&self, num_samples: i64) -> Tensor {
        let latent_dim = self.generator.config().latent_dim;
        let noise = Tensor::randn([num_samples, latent_dim], (tch::Kind::Float, self.device));
        self.generator.generate(&noise)
    }

    /// Generate samples from specific noise vectors
    pub fn generate_from_noise(&self, noise: &Tensor) -> Tensor {
        self.generator.generate(noise)
    }

    /// Discriminate samples (get probability of being real)
    pub fn discriminate(&self, samples: &Tensor) -> Tensor {
        self.discriminator.classify(samples)
    }

    /// Get generator optimizer (Adam with default GAN parameters)
    pub fn gen_optimizer(&self, lr: f64) -> nn::Optimizer {
        nn::Adam {
            beta1: 0.5,
            beta2: 0.999,
            wd: 0.0,
        }
        .build(&self.gen_vs, lr)
        .expect("Failed to create generator optimizer")
    }

    /// Get discriminator optimizer (Adam with default GAN parameters)
    pub fn disc_optimizer(&self, lr: f64) -> nn::Optimizer {
        nn::Adam {
            beta1: 0.5,
            beta2: 0.999,
            wd: 0.0,
        }
        .build(&self.disc_vs, lr)
        .expect("Failed to create discriminator optimizer")
    }

    /// Save model checkpoints
    pub fn save(&self, gen_path: &str, disc_path: &str) -> anyhow::Result<()> {
        self.gen_vs.save(gen_path)?;
        self.disc_vs.save(disc_path)?;
        Ok(())
    }

    /// Load model checkpoints
    pub fn load(&mut self, gen_path: &str, disc_path: &str) -> anyhow::Result<()> {
        self.gen_vs.load(gen_path)?;
        self.disc_vs.load(disc_path)?;
        Ok(())
    }

    /// Get latent dimension
    pub fn latent_dim(&self) -> i64 {
        self.generator.config().latent_dim
    }

    /// Get sequence length
    pub fn sequence_length(&self) -> i64 {
        self.generator.config().sequence_length
    }

    /// Get number of features
    pub fn num_features(&self) -> i64 {
        self.generator.config().num_features
    }

    /// Set model to training mode
    pub fn train(&mut self) {
        self.gen_vs.set_kind(tch::Kind::Float);
        self.disc_vs.set_kind(tch::Kind::Float);
    }

    /// Set model to evaluation mode
    pub fn eval(&mut self) {
        self.gen_vs.freeze();
        self.disc_vs.freeze();
    }

    /// Interpolate between two points in latent space
    ///
    /// Useful for visualizing smooth transitions between generated samples
    ///
    /// # Arguments
    ///
    /// * `z1` - First latent vector
    /// * `z2` - Second latent vector
    /// * `steps` - Number of interpolation steps
    ///
    /// # Returns
    ///
    /// Tensor of shape (steps, sequence_length, num_features)
    pub fn interpolate(&self, z1: &Tensor, z2: &Tensor, steps: i64) -> Tensor {
        let mut samples = Vec::new();

        for i in 0..steps {
            let alpha = i as f64 / (steps - 1) as f64;
            let z = z1 * (1.0 - alpha) + z2 * alpha;
            let sample = self.generator.generate(&z.unsqueeze(0));
            samples.push(sample.squeeze_dim(0));
        }

        Tensor::stack(&samples, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dcgan_creation() {
        let dcgan = DCGAN::with_defaults(24, 5, 100, Device::Cpu);

        assert_eq!(dcgan.latent_dim(), 100);
        assert_eq!(dcgan.sequence_length(), 24);
        assert_eq!(dcgan.num_features(), 5);
    }

    #[test]
    fn test_dcgan_generate() {
        let dcgan = DCGAN::with_defaults(24, 5, 100, Device::Cpu);

        let samples = dcgan.generate(4);
        assert_eq!(samples.size(), vec![4, 24, 5]);
    }

    #[test]
    fn test_dcgan_discriminate() {
        let dcgan = DCGAN::with_defaults(24, 5, 100, Device::Cpu);

        let samples = Tensor::randn([4, 24, 5], (tch::Kind::Float, Device::Cpu));
        let probs = dcgan.discriminate(&samples);

        assert_eq!(probs.size(), vec![4, 1]);
    }

    #[test]
    fn test_dcgan_interpolate() {
        let dcgan = DCGAN::with_defaults(24, 5, 100, Device::Cpu);

        let z1 = Tensor::randn([100], (tch::Kind::Float, Device::Cpu));
        let z2 = Tensor::randn([100], (tch::Kind::Float, Device::Cpu));

        let interpolated = dcgan.interpolate(&z1, &z2, 10);
        assert_eq!(interpolated.size(), vec![10, 24, 5]);
    }
}
