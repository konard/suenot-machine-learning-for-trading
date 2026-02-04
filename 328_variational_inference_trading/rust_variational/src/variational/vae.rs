//! Variational Autoencoder implementation
//!
//! Core VAE model combining encoder and decoder.

use ndarray::Array1;
use rand_distr::{Distribution, Normal};

use super::config::VAEConfig;
use super::decoder::Decoder;
use super::encoder::Encoder;
use super::prediction::Prediction;

/// Variational Autoencoder for Market Data
///
/// Learns latent representations of market states and provides:
/// - Uncertainty-aware return predictions
/// - Market regime detection
/// - Anomaly detection via reconstruction error
#[derive(Debug, Clone)]
pub struct VAE {
    config: VAEConfig,
    encoder: Encoder,
    decoder: Decoder,
}

impl VAE {
    /// Create a new VAE from config
    pub fn new(config: VAEConfig) -> Self {
        let encoder = Encoder::new(config.clone());
        let decoder = Decoder::new(config.clone());

        Self {
            config,
            encoder,
            decoder,
        }
    }

    /// Reparameterization trick
    ///
    /// z = mu + sigma * epsilon, where epsilon ~ N(0, 1)
    pub fn reparameterize(&self, mu: &Array1<f64>, log_var: &Array1<f64>) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let std: Array1<f64> = log_var.mapv(|lv| (0.5 * lv).exp());
        let eps: Array1<f64> = Array1::from_vec(
            (0..mu.len()).map(|_| normal.sample(&mut rng)).collect()
        );

        mu + &std * &eps
    }

    /// Forward pass through VAE
    ///
    /// # Arguments
    /// * `x` - Input features flattened from (seq_len, input_dim)
    ///
    /// # Returns
    /// VAEOutput with all predictions and latent variables
    pub fn forward(&self, x: &Array1<f64>) -> VAEOutput {
        // Encode
        let (mu, log_var) = self.encoder.forward(x);

        // Reparameterize
        let z = self.reparameterize(&mu, &log_var);

        // Decode
        let decoder_output = self.decoder.forward(&z);

        VAEOutput {
            x_recon: decoder_output.x_recon,
            return_mu: decoder_output.return_mu,
            return_logvar: decoder_output.return_logvar,
            regime_logits: decoder_output.regime_logits,
            latent_mu: mu,
            latent_logvar: log_var,
            latent_z: z,
        }
    }

    /// Generate prediction with uncertainty estimation
    ///
    /// Samples multiple times from the latent space to estimate uncertainty.
    pub fn predict(&self, x: &Array1<f64>, n_samples: usize) -> Prediction {
        let (mu, log_var) = self.encoder.forward(x);

        let mut return_samples = Vec::with_capacity(n_samples);
        let mut regime_probs_sum = vec![0.0; self.config.n_regimes];

        for _ in 0..n_samples {
            let z = self.reparameterize(&mu, &log_var);
            let output = self.decoder.forward(&z);

            // Sample from return distribution
            let mut rng = rand::thread_rng();
            let normal = Normal::new(output.return_mu, output.return_std().max(1e-8)).unwrap();
            return_samples.push(normal.sample(&mut rng));

            // Accumulate regime probs
            let probs = output.regime_probs();
            for (i, &prob) in probs.iter().enumerate() {
                regime_probs_sum[i] += prob;
            }
        }

        // Calculate statistics
        let expected_return = return_samples.iter().sum::<f64>() / n_samples as f64;
        let variance = return_samples.iter()
            .map(|&r| (r - expected_return).powi(2))
            .sum::<f64>() / n_samples as f64;
        let return_std = variance.sqrt();

        // Average regime probs
        let regime_probs: Vec<f64> = regime_probs_sum.iter()
            .map(|&p| p / n_samples as f64)
            .collect();

        // Find predicted regime
        let regime = regime_probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Calculate reconstruction error for anomaly detection
        let z = self.reparameterize(&mu, &log_var);
        let output = self.decoder.forward(&z);
        let recon_error: f64 = output.x_recon.iter()
            .zip(x.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>() / x.len() as f64;

        let mut prediction = Prediction::new(
            expected_return,
            return_std,
            regime,
            regime_probs,
            mu.to_vec(),
            log_var.to_vec(),
        );
        prediction.reconstruction_error = Some(recon_error);

        prediction
    }

    /// Compute ELBO loss
    ///
    /// Loss = Reconstruction Loss + beta * KL Divergence
    pub fn compute_loss(
        &self,
        x: &Array1<f64>,
        output: &VAEOutput,
        target_return: Option<f64>,
    ) -> LossOutput {
        // Reconstruction loss (MSE)
        let recon_loss: f64 = output.x_recon.iter()
            .zip(x.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>() / x.len() as f64;

        // KL divergence (analytical for Gaussian)
        // KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        let kl_loss: f64 = -0.5 * output.latent_mu.iter()
            .zip(output.latent_logvar.iter())
            .map(|(&mu, &lv)| 1.0 + lv - mu.powi(2) - lv.exp())
            .sum::<f64>() / output.latent_mu.len() as f64;

        // Return prediction loss (if target provided)
        let return_loss = if let Some(target) = target_return {
            let return_std = (0.5 * output.return_logvar).exp().max(1e-8);
            // Negative log likelihood
            0.5 * ((target - output.return_mu).powi(2) / return_std.powi(2)
                + output.return_logvar
                + (2.0 * std::f64::consts::PI).ln())
        } else {
            0.0
        };

        // Total loss
        let total_loss = recon_loss + return_loss + self.config.beta * kl_loss;

        LossOutput {
            total_loss,
            recon_loss,
            kl_loss,
            return_loss,
        }
    }

    /// Get latent representation
    pub fn encode(&self, x: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        self.encoder.forward(x)
    }

    /// Get config
    pub fn config(&self) -> &VAEConfig {
        &self.config
    }
}

/// Output from VAE forward pass
#[derive(Debug, Clone)]
pub struct VAEOutput {
    /// Reconstructed features
    pub x_recon: Array1<f64>,
    /// Predicted return mean
    pub return_mu: f64,
    /// Predicted return log variance
    pub return_logvar: f64,
    /// Regime classification logits
    pub regime_logits: Array1<f64>,
    /// Latent mean
    pub latent_mu: Array1<f64>,
    /// Latent log variance
    pub latent_logvar: Array1<f64>,
    /// Sampled latent vector
    pub latent_z: Array1<f64>,
}

/// Loss output
#[derive(Debug, Clone)]
pub struct LossOutput {
    /// Total loss
    pub total_loss: f64,
    /// Reconstruction loss
    pub recon_loss: f64,
    /// KL divergence loss
    pub kl_loss: f64,
    /// Return prediction loss
    pub return_loss: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vae_forward() {
        let config = VAEConfig::new(13, 8, vec![64, 32])
            .with_sequence_length(60);

        let vae = VAE::new(config);

        // Create random input
        let input = Array1::from_vec(vec![0.1; 13 * 60]);
        let output = vae.forward(&input);

        assert_eq!(output.x_recon.len(), 13 * 60);
        assert_eq!(output.latent_mu.len(), 8);
        assert_eq!(output.latent_z.len(), 8);
    }

    #[test]
    fn test_vae_predict() {
        let config = VAEConfig::new(13, 8, vec![64, 32])
            .with_sequence_length(60);

        let vae = VAE::new(config);

        let input = Array1::from_vec(vec![0.1; 13 * 60]);
        let prediction = vae.predict(&input, 50);

        assert!(prediction.regime < 3);
        assert_eq!(prediction.regime_probs.len(), 3);
        assert!(prediction.reconstruction_error.is_some());
    }

    #[test]
    fn test_vae_loss() {
        let config = VAEConfig::new(13, 8, vec![64, 32])
            .with_sequence_length(60);

        let vae = VAE::new(config);

        let input = Array1::from_vec(vec![0.1; 13 * 60]);
        let output = vae.forward(&input);
        let loss = vae.compute_loss(&input, &output, Some(0.01));

        assert!(loss.total_loss >= 0.0);
        assert!(loss.recon_loss >= 0.0);
        assert!(loss.kl_loss >= 0.0);
    }
}
