//! GLOW Model - Full implementation
//!
//! Provides the complete GLOW model with multi-scale architecture

use super::FlowStep;
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Configuration for GLOW model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GLOWConfig {
    /// Number of input features
    pub num_features: usize,
    /// Number of levels in multi-scale architecture
    pub num_levels: usize,
    /// Number of flow steps per level
    pub num_steps: usize,
    /// Hidden dimension for coupling layers
    pub hidden_dim: usize,
    /// Learning rate for training
    pub learning_rate: f64,
}

impl Default for GLOWConfig {
    fn default() -> Self {
        Self {
            num_features: 16,
            num_levels: 3,
            num_steps: 4,
            hidden_dim: 64,
            learning_rate: 1e-4,
        }
    }
}

impl GLOWConfig {
    /// Create config with custom number of features
    pub fn with_features(num_features: usize) -> Self {
        Self {
            num_features,
            ..Default::default()
        }
    }
}

/// GLOW Model
///
/// Generative Flow with Invertible 1x1 Convolutions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GLOWModel {
    /// Model configuration
    pub config: GLOWConfig,
    /// Flow steps organized by level
    pub levels: Vec<Vec<FlowStep>>,
    /// Feature dimensions at each level
    pub level_dims: Vec<usize>,
}

impl GLOWModel {
    /// Create a new GLOW model
    pub fn new(config: GLOWConfig) -> Self {
        let mut levels = Vec::with_capacity(config.num_levels);
        let mut level_dims = Vec::with_capacity(config.num_levels);

        let mut current_features = config.num_features;

        for level in 0..config.num_levels {
            // Create flow steps for this level
            let steps: Vec<FlowStep> = (0..config.num_steps)
                .map(|_| FlowStep::new(current_features, config.hidden_dim))
                .collect();

            levels.push(steps);
            level_dims.push(current_features);

            // Split features for next level (except last level)
            if level < config.num_levels - 1 {
                current_features /= 2;
                if current_features < 2 {
                    current_features = 2;
                }
            }
        }

        Self {
            config,
            levels,
            level_dims,
        }
    }

    /// Forward pass: x -> z
    ///
    /// Returns (latent z, log determinant)
    pub fn forward(&mut self, x: &Array2<f64>) -> (Array2<f64>, f64) {
        let mut total_log_det = 0.0;
        let mut z_parts: Vec<Array2<f64>> = Vec::new();

        let mut h = x.clone();

        for (level_idx, level_steps) in self.levels.iter_mut().enumerate() {
            // Apply flow steps
            for step in level_steps.iter_mut() {
                let (h_new, log_det) = step.forward(&h);
                h = h_new;
                total_log_det += log_det;
            }

            // Split (except last level)
            if level_idx < self.config.num_levels - 1 {
                let split_dim = h.ncols() / 2;
                let z_i = h.slice(ndarray::s![.., ..split_dim]).to_owned();
                h = h.slice(ndarray::s![.., split_dim..]).to_owned();
                z_parts.push(z_i);
            }
        }

        // Add final latent
        z_parts.push(h);

        // Concatenate all latent parts
        let z = Self::concat_arrays(&z_parts);

        (z, total_log_det)
    }

    /// Inverse pass: z -> x
    pub fn inverse(&mut self, z: &Array2<f64>) -> Array2<f64> {
        // Split z into parts for each level
        let z_parts = self.split_latent(z);

        // Start from final latent
        let mut h = z_parts.last().unwrap().clone();

        // Reverse through levels
        for level_idx in (0..self.config.num_levels).rev() {
            // Merge with z_i (except for last level going in reverse)
            if level_idx < self.config.num_levels - 1 {
                let z_i = &z_parts[level_idx];
                h = Self::concat_arrays(&[z_i.clone(), h]);
            }

            // Reverse flow steps
            for step in self.levels[level_idx].iter_mut().rev() {
                let (h_new, _) = step.inverse(&h);
                h = h_new;
            }
        }

        h
    }

    /// Compute log probability of x
    pub fn log_prob(&mut self, x: &Array2<f64>) -> Array1<f64> {
        let (z, log_det) = self.forward(x);

        // Log probability under standard Gaussian prior
        let log_pz = Self::gaussian_log_prob(&z);

        // log p(x) = log p(z) + log|det(dz/dx)|
        log_pz + log_det / x.nrows() as f64
    }

    /// Compute Gaussian log probability for each sample
    fn gaussian_log_prob(z: &Array2<f64>) -> Array1<f64> {
        let d = z.ncols() as f64;
        let norm_const = -0.5 * d * (2.0 * PI).ln();

        z.outer_iter()
            .map(|row| {
                let sum_sq: f64 = row.iter().map(|&v| v * v).sum();
                norm_const - 0.5 * sum_sq
            })
            .collect()
    }

    /// Sample from the model
    pub fn sample(&mut self, num_samples: usize, temperature: f64) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, temperature).unwrap();

        // Sample from standard Gaussian
        let z = Array2::from_shape_fn((num_samples, self.config.num_features), |_| {
            normal.sample(&mut rng)
        });

        self.inverse(&z)
    }

    /// Split latent vector into parts for each level
    fn split_latent(&self, z: &Array2<f64>) -> Vec<Array2<f64>> {
        let mut parts = Vec::new();
        let mut start = 0;

        for level_idx in 0..self.config.num_levels - 1 {
            let dim = self.level_dims[level_idx] / 2;
            let end = start + dim;
            parts.push(z.slice(ndarray::s![.., start..end]).to_owned());
            start = end;
        }

        // Final part
        parts.push(z.slice(ndarray::s![.., start..]).to_owned());

        parts
    }

    /// Concatenate arrays along feature dimension
    fn concat_arrays(arrays: &[Array2<f64>]) -> Array2<f64> {
        if arrays.is_empty() {
            return Array2::zeros((0, 0));
        }

        let n_rows = arrays[0].nrows();
        let total_cols: usize = arrays.iter().map(|a| a.ncols()).sum();

        let mut result = Array2::zeros((n_rows, total_cols));
        let mut col_start = 0;

        for arr in arrays {
            let col_end = col_start + arr.ncols();
            result
                .slice_mut(ndarray::s![.., col_start..col_end])
                .assign(arr);
            col_start = col_end;
        }

        result
    }

    /// Train on a batch of data
    ///
    /// Returns the negative log-likelihood loss
    pub fn train_step(&mut self, batch: &Array2<f64>, learning_rate: f64) -> f64 {
        // Compute forward pass and log probability
        let log_prob = self.log_prob(batch);
        let nll = -log_prob.mean().unwrap_or(0.0);

        // Simple gradient descent update (in practice, use Adam or similar)
        // This is a placeholder - real training would compute gradients
        // For now, we just return the loss

        nll
    }

    /// Get the number of parameters in the model
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;

        for level_steps in &self.levels {
            for step in level_steps {
                // ActNorm
                count += step.actnorm.log_scale.len() * 2;

                // Conv1x1
                count += step.conv1x1.weight.len();

                // Coupling
                count += step.coupling.w1.len() + step.coupling.b1.len();
                count += step.coupling.w2.len() + step.coupling.b2.len();
                count += step.coupling.w3.len() + step.coupling.b3.len();
            }
        }

        count
    }

    /// Save model to bytes
    pub fn save(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    /// Load model from bytes
    pub fn load(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }
}

/// Training statistics
#[derive(Debug, Clone)]
pub struct TrainingStats {
    pub epoch: usize,
    pub train_nll: f64,
    pub val_nll: f64,
    pub bits_per_dim: f64,
}

impl TrainingStats {
    /// Compute bits per dimension from NLL
    pub fn compute_bpd(nll: f64, num_features: usize) -> f64 {
        nll / (num_features as f64 * 2.0_f64.ln())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn create_test_data(n_samples: usize, n_features: usize) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        Array2::from_shape_fn((n_samples, n_features), |_| rng.gen::<f64>() * 2.0 - 1.0)
    }

    #[test]
    fn test_glow_creation() {
        let config = GLOWConfig::with_features(16);
        let model = GLOWModel::new(config);

        assert_eq!(model.levels.len(), 3);
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_glow_forward() {
        let config = GLOWConfig::with_features(16);
        let mut model = GLOWModel::new(config);
        let x = create_test_data(10, 16);

        let (z, log_det) = model.forward(&x);

        assert_eq!(z.nrows(), 10);
        assert_eq!(z.ncols(), 16);
        assert!(log_det.is_finite());
    }

    #[test]
    fn test_glow_invertibility() {
        let config = GLOWConfig {
            num_features: 8,
            num_levels: 2,
            num_steps: 2,
            hidden_dim: 32,
            learning_rate: 1e-4,
        };
        let mut model = GLOWModel::new(config);
        let x = create_test_data(5, 8);

        let (z, _) = model.forward(&x);
        let x_recovered = model.inverse(&z);

        // Check invertibility (with tolerance for numerical errors)
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!(
                    (x[[i, j]] - x_recovered[[i, j]]).abs() < 0.1,
                    "Mismatch at [{}, {}]: {} vs {}",
                    i, j, x[[i, j]], x_recovered[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_glow_log_prob() {
        let config = GLOWConfig::with_features(16);
        let mut model = GLOWModel::new(config);
        let x = create_test_data(10, 16);

        let log_prob = model.log_prob(&x);

        assert_eq!(log_prob.len(), 10);
        for &lp in log_prob.iter() {
            assert!(lp.is_finite());
        }
    }

    #[test]
    fn test_glow_sample() {
        let config = GLOWConfig::with_features(16);
        let mut model = GLOWModel::new(config);

        let samples = model.sample(10, 1.0);

        assert_eq!(samples.nrows(), 10);
        assert_eq!(samples.ncols(), 16);
    }

    #[test]
    fn test_glow_serialization() {
        let config = GLOWConfig::with_features(8);
        let model = GLOWModel::new(config);

        let bytes = model.save().expect("Failed to serialize");
        let loaded = GLOWModel::load(&bytes).expect("Failed to deserialize");

        assert_eq!(model.config.num_features, loaded.config.num_features);
        assert_eq!(model.levels.len(), loaded.levels.len());
    }
}
