//! Traits for normalizing flow models.

use anyhow::Result;
use ndarray::Array1;

/// Trait for normalizing flow models
pub trait NormalizingFlow {
    /// Forward pass: latent space -> data space
    fn forward(&self, z: &Array1<f64>) -> Array1<f64>;

    /// Inverse pass: data space -> latent space
    /// Returns (z, log_det_jacobian)
    fn inverse(&self, x: &Array1<f64>) -> (Array1<f64>, f64);

    /// Compute log probability of data
    fn log_prob(&self, x: &Array1<f64>) -> f64;

    /// Generate samples from the learned distribution
    fn sample(&self, n_samples: usize) -> Vec<Array1<f64>>;

    /// Generate samples as a flat vector (for 1D flows)
    fn sample_flat(&self, n_samples: usize) -> Vec<f64> {
        self.sample(n_samples).into_iter().map(|x| x[0]).collect()
    }

    /// Train the flow on data
    fn train(&mut self, data: &[f64], epochs: usize) -> Result<TrainingHistory>;

    /// Get the dimension of the flow
    fn dim(&self) -> usize;
}

/// Training history
#[derive(Debug, Clone, Default)]
pub struct TrainingHistory {
    /// Training loss per epoch
    pub train_loss: Vec<f64>,
    /// Validation loss per epoch
    pub val_loss: Vec<f64>,
    /// Learning rate per epoch
    pub learning_rate: Vec<f64>,
}

impl TrainingHistory {
    /// Get the final training loss
    pub fn final_train_loss(&self) -> Option<f64> {
        self.train_loss.last().copied()
    }

    /// Get the final validation loss
    pub fn final_val_loss(&self) -> Option<f64> {
        self.val_loss.last().copied()
    }

    /// Get the best validation loss
    pub fn best_val_loss(&self) -> Option<f64> {
        self.val_loss.iter().copied().reduce(f64::min)
    }

    /// Get number of epochs trained
    pub fn n_epochs(&self) -> usize {
        self.train_loss.len()
    }
}
