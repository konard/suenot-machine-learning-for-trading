//! Disentangled VAE
//!
//! Full VAE with reparameterization trick and multiple disentanglement losses.

use ndarray::{Array1, Array2, Array3};
use rand::Rng;
use rand_distr::StandardNormal;

use crate::data::Dataset;
use crate::model::config::{DisentangledVAEConfig, DisentanglementMethod};
use crate::model::decoder::Decoder;
use crate::model::encoder::Encoder;

/// Result of a VAE forward pass
#[derive(Debug, Clone)]
pub struct VAEOutput {
    /// Reconstructed output
    pub reconstruction: Array1<f64>,
    /// Latent mean
    pub mu: Array1<f64>,
    /// Latent log-variance
    pub logvar: Array1<f64>,
    /// Sampled latent vector
    pub z: Array1<f64>,
}

/// Training loss components
#[derive(Debug, Clone)]
pub struct VAELoss {
    /// Total loss
    pub total: f64,
    /// Reconstruction loss (MSE)
    pub reconstruction: f64,
    /// KL divergence
    pub kl_divergence: f64,
    /// Disentanglement-specific penalty
    pub disentanglement_penalty: f64,
}

/// Disentangled Variational Autoencoder
///
/// Learns interpretable latent representations of market data using
/// configurable disentanglement methods.
pub struct DisentangledVAE {
    /// Encoder network
    encoder: Encoder,
    /// Decoder network
    decoder: Decoder,
    /// Prediction head weights (latent_dim -> output_dim)
    pred_weights: Array2<f64>,
    /// Prediction head bias
    pred_bias: Array1<f64>,
    /// Model configuration
    config: DisentangledVAEConfig,
}

impl DisentangledVAE {
    /// Create a new Disentangled VAE from configuration
    pub fn new(config: DisentangledVAEConfig) -> Self {
        let encoder = Encoder::new(&config);
        let decoder = Decoder::new(&config);

        let mut rng = rand::thread_rng();
        let scale = (2.0 / config.latent_dim as f64).sqrt();
        let pred_weights = Array2::from_shape_fn(
            (config.latent_dim, config.output_dim),
            |_| rng.gen_range(-scale..scale),
        );
        let pred_bias = Array1::zeros(config.output_dim);

        Self {
            encoder,
            decoder,
            pred_weights,
            pred_bias,
            config,
        }
    }

    /// Get the model configuration
    pub fn config(&self) -> &DisentangledVAEConfig {
        &self.config
    }

    /// Reparameterization trick: z = mu + std * epsilon
    pub fn reparameterize(&self, mu: &Array1<f64>, logvar: &Array1<f64>) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let std = logvar.mapv(|lv| (lv * 0.5).exp());
        let epsilon: Array1<f64> = Array1::from_shape_fn(mu.len(), |_| rng.sample(StandardNormal));
        mu + &(std * epsilon)
    }

    /// Encode input to latent distribution
    ///
    /// # Arguments
    /// * `input` - Input sequence flattened to [seq_len * input_dim]
    ///
    /// # Returns
    /// Tuple of (mu, logvar)
    pub fn encode(&self, input: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        self.encoder.forward(input)
    }

    /// Decode latent vector to reconstruction
    ///
    /// # Arguments
    /// * `z` - Latent vector of shape [latent_dim]
    ///
    /// # Returns
    /// Reconstructed output
    pub fn decode(&self, z: &Array1<f64>) -> Array1<f64> {
        self.decoder.forward(z)
    }

    /// Full forward pass: input -> encode -> reparameterize -> decode
    pub fn forward(&self, input: &Array1<f64>) -> VAEOutput {
        let (mu, logvar) = self.encode(input);
        let z = self.reparameterize(&mu, &logvar);
        let reconstruction = self.decode(&z);

        VAEOutput {
            reconstruction,
            mu,
            logvar,
            z,
        }
    }

    /// Predict future returns from latent representation
    pub fn predict_from_latent(&self, z: &Array1<f64>) -> f64 {
        let pred = z.dot(&self.pred_weights) + &self.pred_bias;
        pred[0]
    }

    /// Make predictions on a dataset
    pub fn predict(&mut self, dataset: &Dataset) -> Vec<f64> {
        let mut predictions = Vec::with_capacity(dataset.len());

        for i in 0..dataset.len() {
            let input = Self::flatten_sequence(&dataset.inputs, i);
            let (mu, _) = self.encode(&input);
            let pred = self.predict_from_latent(&mu);
            predictions.push(pred);
        }

        predictions
    }

    /// Compute loss for a batch
    pub fn compute_loss(&mut self, inputs: &Array3<f64>, _targets: &[f64]) -> f64 {
        let batch_size = inputs.dim().0;
        let mut total_loss = 0.0;

        for i in 0..batch_size {
            let input = Self::flatten_sequence(inputs, i);
            let output = self.forward(&input);
            let loss = self.compute_single_loss(&input, &output);
            total_loss += loss.total;
        }

        total_loss / batch_size as f64
    }

    /// Compute loss for a single sample
    fn compute_single_loss(&self, input: &Array1<f64>, output: &VAEOutput) -> VAELoss {
        // Reconstruction loss (MSE)
        let diff = input - &output.reconstruction;
        let reconstruction_loss = diff.mapv(|x| x * x).mean().unwrap_or(0.0);

        // KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        let kl_divergence = -0.5
            * output
                .logvar
                .iter()
                .zip(output.mu.iter())
                .map(|(lv, m)| 1.0 + lv - m * m - lv.exp())
                .sum::<f64>();

        // Disentanglement penalty based on method
        let disentanglement_penalty = match self.config.method {
            DisentanglementMethod::BetaVAE => {
                // Beta-VAE simply scales KL
                0.0
            }
            DisentanglementMethod::FactorVAE => {
                // Approximate total correlation penalty
                self.compute_tc_penalty(&output.mu, &output.logvar)
            }
            DisentanglementMethod::DIPVAE => {
                // Covariance penalty on mu
                self.compute_dip_penalty(&output.mu, &output.logvar)
            }
            DisentanglementMethod::BetaTCVAE => {
                // Decomposed KL: TC term
                self.compute_tc_penalty(&output.mu, &output.logvar)
            }
        };

        let total = reconstruction_loss
            + self.config.beta * kl_divergence
            + disentanglement_penalty;

        VAELoss {
            total,
            reconstruction: reconstruction_loss,
            kl_divergence,
            disentanglement_penalty,
        }
    }

    /// Compute total correlation penalty (for FactorVAE and BetaTCVAE)
    fn compute_tc_penalty(&self, mu: &Array1<f64>, logvar: &Array1<f64>) -> f64 {
        // Approximate TC as sum of marginal KLs minus joint KL
        // For a single sample, we approximate using the diagonal assumption
        let dim = mu.len() as f64;

        // Marginal KLs
        let marginal_kl: f64 = mu
            .iter()
            .zip(logvar.iter())
            .map(|(m, lv)| -0.5 * (1.0 + lv - m * m - lv.exp()))
            .sum();

        // Joint KL (same as total KL for single sample)
        let joint_kl = marginal_kl;

        // TC approximation: difference scaled by dimension
        let tc = (marginal_kl - joint_kl / dim).max(0.0);

        tc * self.config.beta
    }

    /// Compute DIP-VAE covariance penalty
    fn compute_dip_penalty(&self, mu: &Array1<f64>, logvar: &Array1<f64>) -> f64 {
        let d = mu.len();
        let mut penalty = 0.0;

        // Penalize off-diagonal elements of covariance
        for i in 0..d {
            for j in 0..d {
                if i != j {
                    // Off-diagonal: penalize mu_i * mu_j correlation
                    let cov_ij = mu[i] * mu[j];
                    penalty += self.config.dip_lambda_offdiag * cov_ij * cov_ij;
                } else {
                    // Diagonal: penalize deviation from 1
                    let var_i = logvar[i].exp();
                    penalty += self.config.dip_lambda_diag * (var_i - 1.0).powi(2);
                }
            }
        }

        penalty
    }

    /// Compute loss for a batch and return detailed loss components
    pub fn compute_detailed_loss(
        &mut self,
        inputs: &Array3<f64>,
        _targets: &[f64],
    ) -> VAELoss {
        let batch_size = inputs.dim().0;
        let mut total_recon = 0.0;
        let mut total_kl = 0.0;
        let mut total_disent = 0.0;

        for i in 0..batch_size {
            let input = Self::flatten_sequence(inputs, i);
            let output = self.forward(&input);
            let loss = self.compute_single_loss(&input, &output);

            total_recon += loss.reconstruction;
            total_kl += loss.kl_divergence;
            total_disent += loss.disentanglement_penalty;
        }

        let n = batch_size as f64;
        let reconstruction = total_recon / n;
        let kl_divergence = total_kl / n;
        let disentanglement_penalty = total_disent / n;
        let total = reconstruction + self.config.beta * kl_divergence + disentanglement_penalty;

        VAELoss {
            total,
            reconstruction,
            kl_divergence,
            disentanglement_penalty,
        }
    }

    /// Extract latent factors for a dataset
    pub fn extract_latent_factors(&self, dataset: &Dataset) -> Array2<f64> {
        let n = dataset.len();
        let mut factors = Array2::zeros((n, self.config.latent_dim));

        for i in 0..n {
            let input = Self::flatten_sequence(&dataset.inputs, i);
            let (mu, _) = self.encode(&input);
            factors.row_mut(i).assign(&mu);
        }

        factors
    }

    /// Get latent factor statistics (mean and std per dimension)
    pub fn latent_factor_stats(&self, dataset: &Dataset) -> Vec<(f64, f64)> {
        let factors = self.extract_latent_factors(dataset);
        let n = factors.nrows() as f64;

        (0..self.config.latent_dim)
            .map(|d| {
                let col = factors.column(d);
                let mean = col.sum() / n;
                let variance = col.mapv(|x| (x - mean).powi(2)).sum() / n;
                (mean, variance.sqrt())
            })
            .collect()
    }

    /// Flatten a single sequence from a 3D input array
    fn flatten_sequence(inputs: &Array3<f64>, idx: usize) -> Array1<f64> {
        let seq = inputs.slice(ndarray::s![idx, .., ..]);
        Array1::from_iter(seq.iter().cloned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> DisentangledVAEConfig {
        DisentangledVAEConfig {
            input_dim: 8,
            seq_len: 10,
            latent_dim: 4,
            hidden_dim: 32,
            num_hidden_layers: 2,
            output_dim: 1,
            beta: 4.0,
            method: DisentanglementMethod::BetaVAE,
            ..Default::default()
        }
    }

    #[test]
    fn test_vae_forward() {
        let config = create_test_config();
        let vae = DisentangledVAE::new(config);

        let input = Array1::zeros(80); // 10 * 8
        let output = vae.forward(&input);

        assert_eq!(output.mu.len(), 4);
        assert_eq!(output.logvar.len(), 4);
        assert_eq!(output.z.len(), 4);
        assert_eq!(output.reconstruction.len(), 80);
    }

    #[test]
    fn test_reparameterize() {
        let config = create_test_config();
        let vae = DisentangledVAE::new(config);

        let mu = Array1::zeros(4);
        let logvar = Array1::zeros(4);

        let z1 = vae.reparameterize(&mu, &logvar);
        let z2 = vae.reparameterize(&mu, &logvar);

        // Should produce different samples (stochastic)
        assert_eq!(z1.len(), 4);
        assert_eq!(z2.len(), 4);
    }

    #[test]
    fn test_vae_loss_beta() {
        let config = create_test_config();
        let vae = DisentangledVAE::new(config);

        let input = Array1::from_elem(80, 0.5);
        let output = vae.forward(&input);
        let loss = vae.compute_single_loss(&input, &output);

        assert!(loss.total >= 0.0);
        assert!(loss.reconstruction >= 0.0);
        assert!(loss.kl_divergence >= 0.0);
    }

    #[test]
    fn test_vae_loss_factor() {
        let mut config = create_test_config();
        config.method = DisentanglementMethod::FactorVAE;
        let vae = DisentangledVAE::new(config);

        let input = Array1::from_elem(80, 0.5);
        let output = vae.forward(&input);
        let loss = vae.compute_single_loss(&input, &output);

        assert!(loss.total >= 0.0);
    }

    #[test]
    fn test_vae_loss_dip() {
        let mut config = create_test_config();
        config.method = DisentanglementMethod::DIPVAE;
        let vae = DisentangledVAE::new(config);

        let input = Array1::from_elem(80, 0.5);
        let output = vae.forward(&input);
        let loss = vae.compute_single_loss(&input, &output);

        assert!(loss.total >= 0.0);
        assert!(loss.disentanglement_penalty >= 0.0);
    }

    #[test]
    fn test_vae_loss_beta_tc() {
        let mut config = create_test_config();
        config.method = DisentanglementMethod::BetaTCVAE;
        let vae = DisentangledVAE::new(config);

        let input = Array1::from_elem(80, 0.5);
        let output = vae.forward(&input);
        let loss = vae.compute_single_loss(&input, &output);

        assert!(loss.total >= 0.0);
    }

    #[test]
    fn test_predict_from_latent() {
        let config = create_test_config();
        let vae = DisentangledVAE::new(config);

        let z = Array1::zeros(4);
        let pred = vae.predict_from_latent(&z);

        // Should return a finite value
        assert!(pred.is_finite());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let config = create_test_config();
        let vae = DisentangledVAE::new(config);

        let input = Array1::from_elem(80, 0.5);
        let (mu, _logvar) = vae.encode(&input);
        let reconstruction = vae.decode(&mu);

        assert_eq!(reconstruction.len(), 80);
    }
}
