//! # RealNVP Network
//!
//! Implementation of the full RealNVP normalizing flow model.
//!
//! RealNVP is composed of alternating affine coupling layers with different masks,
//! optionally interspersed with activation normalization layers.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::{Normal, Distribution};
use std::f64::consts::PI;

use super::coupling::{CouplingLayer, MaskType};
use super::actnorm::ActNorm;
use crate::config;

/// RealNVP Normalizing Flow Model
///
/// Composed of alternating coupling layers that transform data
/// from complex distribution to simple Gaussian and back.
#[derive(Debug)]
pub struct RealNVP {
    /// Input dimension
    dim: usize,
    /// Coupling layers
    coupling_layers: Vec<CouplingLayer>,
    /// Optional ActNorm layers
    actnorm_layers: Vec<ActNorm>,
    /// Whether to use ActNorm
    use_actnorm: bool,
}

impl RealNVP {
    /// Create a new RealNVP flow
    ///
    /// # Arguments
    ///
    /// * `dim` - Input dimension
    /// * `hidden_dim` - Hidden dimension for coupling networks
    /// * `num_coupling_layers` - Number of coupling layers
    pub fn new(dim: usize, hidden_dim: usize, num_coupling_layers: usize) -> Self {
        Self::with_config(dim, hidden_dim, num_coupling_layers, config::DEFAULT_NUM_HIDDEN_LAYERS, false)
    }

    /// Create a new RealNVP flow with ActNorm
    pub fn with_actnorm(dim: usize, hidden_dim: usize, num_coupling_layers: usize) -> Self {
        Self::with_config(dim, hidden_dim, num_coupling_layers, config::DEFAULT_NUM_HIDDEN_LAYERS, true)
    }

    /// Create a new RealNVP flow with full configuration
    pub fn with_config(
        dim: usize,
        hidden_dim: usize,
        num_coupling_layers: usize,
        num_hidden_layers: usize,
        use_actnorm: bool,
    ) -> Self {
        let mut coupling_layers = Vec::with_capacity(num_coupling_layers);
        let mut actnorm_layers = Vec::new();

        for i in 0..num_coupling_layers {
            // Alternate between even and odd masks
            let mask_type = if i % 2 == 0 { MaskType::Even } else { MaskType::Odd };

            if use_actnorm {
                actnorm_layers.push(ActNorm::new(dim));
            }

            coupling_layers.push(CouplingLayer::new(
                dim,
                hidden_dim,
                num_hidden_layers,
                mask_type,
            ));
        }

        Self {
            dim,
            coupling_layers,
            actnorm_layers,
            use_actnorm,
        }
    }

    /// Forward pass: x -> z
    ///
    /// Transforms data from complex distribution to Gaussian.
    /// Returns (z, log_det) where log_det is the total log determinant.
    pub fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, f64) {
        let mut z = x.clone();
        let mut total_log_det = 0.0;

        for (i, coupling) in self.coupling_layers.iter().enumerate() {
            // ActNorm if enabled
            if self.use_actnorm && i < self.actnorm_layers.len() {
                let (z_new, log_det) = self.actnorm_layers[i].forward(&z);
                z = z_new;
                total_log_det += log_det;
            }

            // Coupling layer
            let (z_new, log_det) = coupling.forward(&z);
            z = z_new;
            total_log_det += log_det;
        }

        (z, total_log_det)
    }

    /// Inverse pass: z -> x
    ///
    /// Transforms data from Gaussian back to original distribution.
    pub fn inverse(&self, z: &Array1<f64>) -> Array1<f64> {
        let mut x = z.clone();

        // Iterate in reverse order
        for (i, coupling) in self.coupling_layers.iter().enumerate().rev() {
            // Inverse coupling layer
            x = coupling.inverse(&x);

            // Inverse ActNorm if enabled
            if self.use_actnorm && i < self.actnorm_layers.len() {
                x = self.actnorm_layers[i].inverse(&x);
            }
        }

        x
    }

    /// Compute log probability of x
    ///
    /// Uses the change of variables formula:
    /// log p(x) = log p(z) + log |det(df/dx)|
    pub fn log_prob(&self, x: &Array1<f64>) -> f64 {
        let (z, log_det) = self.forward(x);

        // Log probability under standard Gaussian
        let log_pz: f64 = z.iter()
            .map(|&zi| -0.5 * (zi * zi + (2.0 * PI).ln()))
            .sum();

        // Change of variables
        log_pz + log_det
    }

    /// Forward pass for batch
    pub fn forward_batch(&mut self, x: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
        let batch_size = x.nrows();
        let mut z = x.clone();
        let mut total_log_det = Array1::zeros(batch_size);

        for (i, coupling) in self.coupling_layers.iter().enumerate() {
            // ActNorm if enabled
            if self.use_actnorm && i < self.actnorm_layers.len() {
                let (z_new, log_det) = self.actnorm_layers[i].forward_batch(&z);
                z = z_new;
                total_log_det = total_log_det + log_det;
            }

            // Coupling layer
            let (z_new, log_det) = coupling.forward_batch(&z);
            z = z_new;
            total_log_det = total_log_det + log_det;
        }

        (z, total_log_det)
    }

    /// Inverse pass for batch
    pub fn inverse_batch(&self, z: &Array2<f64>) -> Array2<f64> {
        let mut x = z.clone();

        for (i, coupling) in self.coupling_layers.iter().enumerate().rev() {
            x = coupling.inverse_batch(&x);

            if self.use_actnorm && i < self.actnorm_layers.len() {
                x = self.actnorm_layers[i].inverse_batch(&x);
            }
        }

        x
    }

    /// Compute log probability for batch
    pub fn log_prob_batch(&mut self, x: &Array2<f64>) -> Array1<f64> {
        let (z, log_det) = self.forward_batch(x);

        // Log probability under standard Gaussian for each sample
        let log_pz: Array1<f64> = z.rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .map(|&zi| -0.5 * (zi * zi + (2.0 * PI).ln()))
                    .sum()
            })
            .collect();

        log_pz + log_det
    }

    /// Sample from the learned distribution
    pub fn sample(&self, n_samples: usize) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Sample from standard Gaussian
        let z = Array2::from_shape_fn((n_samples, self.dim), |_| normal.sample(&mut rng));

        // Transform to data space
        self.inverse_batch(&z)
    }

    /// Sample a single point
    pub fn sample_one(&self) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let z: Array1<f64> = Array1::from_shape_fn(self.dim, |_| normal.sample(&mut rng));
        self.inverse(&z)
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get number of coupling layers
    pub fn num_layers(&self) -> usize {
        self.coupling_layers.len()
    }

    /// Get all parameters
    pub fn parameters(&self) -> Vec<f64> {
        let mut params = Vec::new();

        for layer in &self.coupling_layers {
            params.extend(layer.parameters());
        }

        for layer in &self.actnorm_layers {
            params.extend(layer.parameters());
        }

        params
    }

    /// Set parameters
    pub fn set_parameters(&mut self, params: &[f64]) {
        let mut idx = 0;

        for layer in &mut self.coupling_layers {
            let n_params = layer.parameters().len();
            layer.set_parameters(&params[idx..idx + n_params]);
            idx += n_params;
        }

        for layer in &mut self.actnorm_layers {
            let n_params = layer.parameters().len();
            layer.set_parameters(&params[idx..idx + n_params]);
            idx += n_params;
        }
    }

    /// Compute negative log likelihood loss for training
    pub fn nll_loss(&mut self, x: &Array2<f64>) -> f64 {
        let log_probs = self.log_prob_batch(x);
        -log_probs.mean().unwrap_or(0.0)
    }

    /// Estimate density at a point (exp of log_prob, but clamped for safety)
    pub fn density(&self, x: &Array1<f64>) -> f64 {
        let log_p = self.log_prob(x);
        // Clamp to avoid numerical issues
        log_p.clamp(-100.0, 50.0).exp()
    }

    /// Check if a point is "in distribution" based on log probability threshold
    pub fn is_in_distribution(&self, x: &Array1<f64>, threshold: f64) -> bool {
        self.log_prob(x) > threshold
    }
}

/// Training utilities for RealNVP
pub struct RealNVPTrainer {
    /// Learning rate
    learning_rate: f64,
    /// Weight decay
    weight_decay: f64,
    /// Best validation loss seen
    best_val_loss: f64,
    /// Best parameters
    best_params: Option<Vec<f64>>,
}

impl RealNVPTrainer {
    /// Create a new trainer
    pub fn new(learning_rate: f64, weight_decay: f64) -> Self {
        Self {
            learning_rate,
            weight_decay,
            best_val_loss: f64::INFINITY,
            best_params: None,
        }
    }

    /// Train for one epoch
    pub fn train_epoch(
        &mut self,
        model: &mut RealNVP,
        train_data: &Array2<f64>,
        batch_size: usize,
    ) -> f64 {
        let n_samples = train_data.nrows();
        let mut total_loss = 0.0;
        let mut n_batches = 0;

        // Simple SGD without shuffling for now
        for i in (0..n_samples).step_by(batch_size) {
            let end = (i + batch_size).min(n_samples);
            let batch = train_data.slice(ndarray::s![i..end, ..]).to_owned();

            let loss = model.nll_loss(&batch);
            total_loss += loss;
            n_batches += 1;

            // Note: In a real implementation, you would compute gradients
            // and update parameters here. This is a simplified version.
        }

        total_loss / n_batches as f64
    }

    /// Validate the model
    pub fn validate(&mut self, model: &mut RealNVP, val_data: &Array2<f64>) -> f64 {
        let loss = model.nll_loss(val_data);

        if loss < self.best_val_loss {
            self.best_val_loss = loss;
            self.best_params = Some(model.parameters());
        }

        loss
    }

    /// Restore best parameters
    pub fn restore_best(&self, model: &mut RealNVP) {
        if let Some(ref params) = self.best_params {
            model.set_parameters(params);
        }
    }

    /// Get best validation loss
    pub fn best_loss(&self) -> f64 {
        self.best_val_loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_realnvp_forward_inverse() {
        let flow = RealNVP::new(4, 32, 4);

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let (z, _) = flow.forward(&x);
        let x_reconstructed = flow.inverse(&z);

        for (a, b) in x.iter().zip(x_reconstructed.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_realnvp_log_prob() {
        let flow = RealNVP::new(4, 32, 4);

        let x = Array1::from_vec(vec![0.5, -0.5, 1.0, -1.0]);
        let log_prob = flow.log_prob(&x);

        assert!(log_prob.is_finite());
    }

    #[test]
    fn test_realnvp_sample() {
        let flow = RealNVP::new(4, 32, 4);

        let samples = flow.sample(10);

        assert_eq!(samples.nrows(), 10);
        assert_eq!(samples.ncols(), 4);

        // Check all values are finite
        for &val in samples.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_realnvp_batch() {
        let mut flow = RealNVP::new(4, 32, 4);

        let x = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5, -1.0, 0.0, 1.0, 2.0],
        ).unwrap();

        let (z, log_det) = flow.forward_batch(&x);
        let x_reconstructed = flow.inverse_batch(&z);

        assert_eq!(z.nrows(), 3);
        assert_eq!(z.ncols(), 4);
        assert_eq!(log_det.len(), 3);

        for (a, b) in x.iter().zip(x_reconstructed.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_realnvp_with_actnorm() {
        let mut flow = RealNVP::with_actnorm(4, 32, 4);

        // Initialize with some data
        let init_data = Array2::from_shape_fn((100, 4), |(i, j)| (i + j) as f64 / 10.0);
        let _ = flow.forward_batch(&init_data);

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let (z, _) = flow.forward(&x);
        let x_reconstructed = flow.inverse(&z);

        for (a, b) in x.iter().zip(x_reconstructed.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
    }
}
