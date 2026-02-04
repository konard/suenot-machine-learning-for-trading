//! Complete normalizing flow model

use crate::flow::config::FlowConfig;
use crate::flow::layers::{ActNorm, AffineCoupling, Permutation};
use anyhow::Result;
use ndarray::{Array1, Array2, Axis};
use std::f64::consts::PI;

/// Layer type enum for the flow model
#[derive(Debug, Clone)]
pub enum FlowLayerType {
    ActNorm(ActNorm),
    Coupling(AffineCoupling),
    Permutation(Permutation),
}

/// Complete Normalizing Flow model
#[derive(Debug, Clone)]
pub struct NormalizingFlow {
    pub config: FlowConfig,
    pub layers: Vec<FlowLayerType>,
}

impl NormalizingFlow {
    /// Create new flow model from configuration
    pub fn new(config: FlowConfig) -> Self {
        let mut layers = Vec::new();

        for i in 0..config.num_layers {
            // ActNorm
            layers.push(FlowLayerType::ActNorm(ActNorm::new(config.input_dim)));

            // Affine Coupling
            layers.push(FlowLayerType::Coupling(AffineCoupling::new(
                config.input_dim,
                config.hidden_dim,
            )));

            // Permutation (except last layer)
            if i < config.num_layers - 1 {
                layers.push(FlowLayerType::Permutation(Permutation::new(config.input_dim)));
            }
        }

        Self { config, layers }
    }

    /// Forward pass: transform data to latent space
    pub fn forward(&mut self, x: &Array2<f64>) -> (Array2<f64>, f64) {
        let mut z = x.clone();
        let mut log_det_total = 0.0;

        for layer in &mut self.layers {
            let (z_new, log_det) = match layer {
                FlowLayerType::ActNorm(l) => l.forward(&z),
                FlowLayerType::Coupling(l) => l.forward(&z),
                FlowLayerType::Permutation(l) => l.forward(&z),
            };
            z = z_new;
            log_det_total += log_det;
        }

        (z, log_det_total)
    }

    /// Inverse pass: transform latent to data space
    pub fn inverse(&self, z: &Array2<f64>) -> Array2<f64> {
        let mut x = z.clone();

        for layer in self.layers.iter().rev() {
            x = match layer {
                FlowLayerType::ActNorm(l) => l.inverse(&x),
                FlowLayerType::Coupling(l) => l.inverse(&x),
                FlowLayerType::Permutation(l) => l.inverse(&x),
            };
        }

        x
    }

    /// Compute log probability of data
    pub fn log_prob(&mut self, x: &Array2<f64>) -> Array1<f64> {
        let (z, log_det) = self.forward(x);
        let batch_size = x.nrows();
        let dim = self.config.input_dim as f64;

        // Log probability under standard Gaussian
        let log_pz: Array1<f64> = z
            .rows()
            .into_iter()
            .map(|row| {
                let z_sq_sum: f64 = row.iter().map(|v| v * v).sum();
                -0.5 * (dim * (2.0 * PI).ln() + z_sq_sum)
            })
            .collect();

        // Apply change of variables: log p(x) = log p(z) + log |det J|
        let log_det_per_sample = log_det / batch_size as f64;
        log_pz.mapv(|lp| lp + log_det_per_sample)
    }

    /// Compute negative log-likelihood loss
    pub fn nll_loss(&mut self, x: &Array2<f64>) -> f64 {
        let log_probs = self.log_prob(x);
        -log_probs.mean().unwrap_or(0.0)
    }

    /// Generate samples from the model
    pub fn sample(&self, num_samples: usize) -> Array2<f64> {
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::thread_rng();

        // Sample from standard Gaussian
        let z = Array2::from_shape_fn((num_samples, self.config.input_dim), |_| {
            normal.sample(&mut rng)
        });

        // Transform to data space
        self.inverse(&z)
    }

    /// Train the model on data
    pub fn train(&mut self, data: &Array2<f64>, epochs: usize) -> Result<Vec<f64>> {
        let mut losses = Vec::new();
        let batch_size = self.config.batch_size.min(data.nrows());

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            // Simple batching (shuffle would be better in production)
            for start in (0..data.nrows()).step_by(batch_size) {
                let end = (start + batch_size).min(data.nrows());
                let batch = data.slice(ndarray::s![start..end, ..]).to_owned();

                let loss = self.nll_loss(&batch);
                epoch_loss += loss;
                num_batches += 1;

                // Note: Actual gradient update would require autograd
                // This is a simplified demonstration
            }

            let avg_loss = epoch_loss / num_batches as f64;
            losses.push(avg_loss);

            if epoch % 10 == 0 {
                log::info!("Epoch {}: loss = {:.4}", epoch, avg_loss);
            }
        }

        Ok(losses)
    }

    /// Get model dimension
    pub fn dim(&self) -> usize {
        self.config.input_dim
    }
}

impl Default for NormalizingFlow {
    fn default() -> Self {
        Self::new(FlowConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_flow_invertible() {
        let config = FlowConfig::default().with_input_dim(4).with_num_layers(2);
        let mut flow = NormalizingFlow::new(config);

        let x = Array2::from_shape_fn((5, 4), |(i, j)| ((i + j) as f64) * 0.1);

        // Initialize by running forward
        let (z, _) = flow.forward(&x);
        let x_recon = flow.inverse(&z);

        // Check reconstruction
        for i in 0..5 {
            for j in 0..4 {
                assert_abs_diff_eq!(x[[i, j]], x_recon[[i, j]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_log_prob() {
        let config = FlowConfig::default().with_input_dim(4).with_num_layers(2);
        let mut flow = NormalizingFlow::new(config);

        let x = Array2::from_shape_fn((10, 4), |(i, j)| ((i + j) as f64) * 0.01);
        let log_probs = flow.log_prob(&x);

        assert_eq!(log_probs.len(), 10);
        // Log probs should be finite
        assert!(log_probs.iter().all(|&lp| lp.is_finite()));
    }

    #[test]
    fn test_sampling() {
        let config = FlowConfig::default().with_input_dim(4).with_num_layers(2);
        let flow = NormalizingFlow::new(config);

        let samples = flow.sample(100);
        assert_eq!(samples.shape(), &[100, 4]);
    }
}
