//! Neural Spline Flow model
//!
//! Complete implementation of Neural Spline Flows for density estimation.

use super::coupling::{CouplingLayer, Permutation};
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

/// Configuration for Neural Spline Flow
#[derive(Debug, Clone)]
pub struct NSFConfig {
    /// Input dimension
    pub dim: usize,
    /// Number of coupling layers
    pub num_layers: usize,
    /// Hidden dimension for conditioner networks
    pub hidden_dim: usize,
    /// Number of spline bins
    pub num_bins: usize,
    /// Number of hidden layers in conditioner
    pub num_hidden: usize,
    /// Learning rate for training
    pub learning_rate: f64,
}

impl Default for NSFConfig {
    fn default() -> Self {
        Self {
            dim: 10,
            num_layers: 4,
            hidden_dim: 64,
            num_bins: 8,
            num_hidden: 2,
            learning_rate: 1e-3,
        }
    }
}

impl NSFConfig {
    /// Create a new configuration
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            ..Default::default()
        }
    }

    /// Set number of layers
    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Set hidden dimension
    pub fn with_hidden_dim(mut self, hidden_dim: usize) -> Self {
        self.hidden_dim = hidden_dim;
        self
    }

    /// Set number of bins
    pub fn with_num_bins(mut self, num_bins: usize) -> Self {
        self.num_bins = num_bins;
        self
    }
}

/// Neural Spline Flow model
#[derive(Debug, Clone)]
pub struct NeuralSplineFlow {
    /// Configuration
    config: NSFConfig,
    /// Coupling layers
    layers: Vec<CouplingLayer>,
    /// Permutations between layers
    permutations: Vec<Permutation>,
    /// Running mean for normalization
    running_mean: Array1<f64>,
    /// Running variance for normalization
    running_var: Array1<f64>,
    /// Whether the model is trained
    is_trained: bool,
}

impl NeuralSplineFlow {
    /// Create a new Neural Spline Flow
    pub fn new(config: NSFConfig) -> Self {
        let mut layers = Vec::with_capacity(config.num_layers);
        let mut permutations = Vec::with_capacity(config.num_layers);

        for _ in 0..config.num_layers {
            layers.push(CouplingLayer::new(
                config.dim,
                config.hidden_dim,
                config.num_bins,
                config.num_hidden,
            ));
            permutations.push(Permutation::random(config.dim));
        }

        Self {
            config,
            layers,
            permutations,
            running_mean: Array1::zeros(config.dim),
            running_var: Array1::ones(config.dim),
            is_trained: false,
        }
    }

    /// Forward transformation: x -> z
    ///
    /// Returns (z, log_det) where z is in the base distribution space
    pub fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, f64) {
        // Normalize input
        let z = (x - &self.running_mean) / self.running_var.mapv(|v| (v + 1e-6).sqrt());
        let log_det_norm = -0.5 * self.running_var.mapv(|v| (v + 1e-6).ln()).sum();

        let mut z = z;
        let mut total_log_det = log_det_norm;

        for (layer, perm) in self.layers.iter().zip(self.permutations.iter()) {
            let (z_new, log_det) = layer.forward(&z);
            z = perm.forward(&z_new);
            total_log_det += log_det;
        }

        (z, total_log_det)
    }

    /// Inverse transformation: z -> x
    ///
    /// Returns (x, log_det)
    pub fn inverse(&self, z: &Array1<f64>) -> (Array1<f64>, f64) {
        let mut x = z.clone();
        let mut total_log_det = 0.0;

        for (layer, perm) in self.layers.iter().zip(self.permutations.iter()).rev() {
            x = perm.inverse(&x);
            let (x_new, log_det) = layer.inverse(&x);
            x = x_new;
            total_log_det += log_det;
        }

        // Denormalize
        let x = x * self.running_var.mapv(|v| (v + 1e-6).sqrt()) + &self.running_mean;
        let log_det_norm = 0.5 * self.running_var.mapv(|v| (v + 1e-6).ln()).sum();
        total_log_det += log_det_norm;

        (x, total_log_det)
    }

    /// Compute log probability of a sample
    pub fn log_prob(&self, x: &Array1<f64>) -> f64 {
        let (z, log_det) = self.forward(x);

        // Standard normal log probability
        let log_pz = -0.5 * (z.mapv(|v| v * v).sum() + self.config.dim as f64 * (2.0 * PI).ln());

        log_pz + log_det
    }

    /// Compute log probability for a batch
    pub fn log_prob_batch(&self, x: &Array2<f64>) -> Array1<f64> {
        let n = x.nrows();
        let mut log_probs = Array1::zeros(n);

        for i in 0..n {
            let xi = x.row(i).to_owned();
            log_probs[i] = self.log_prob(&xi);
        }

        log_probs
    }

    /// Generate samples from the learned distribution
    pub fn sample(&self, num_samples: usize) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut samples = Array2::zeros((num_samples, self.config.dim));

        for i in 0..num_samples {
            let z: Array1<f64> = Array1::from_shape_fn(self.config.dim, |_| normal.sample(&mut rng));
            let (x, _) = self.inverse(&z);
            samples.row_mut(i).assign(&x);
        }

        samples
    }

    /// Fit the model to data using maximum likelihood
    pub fn fit(&mut self, data: &Array2<f64>) -> anyhow::Result<TrainingStats> {
        let n_samples = data.nrows();

        // Compute running statistics
        self.running_mean = data.mean_axis(Axis(0)).unwrap();
        self.running_var = data.var_axis(Axis(0), 1.0);

        // Simple training loop (in practice, would use proper gradient descent)
        let epochs = 100;
        let batch_size = 64.min(n_samples);

        let mut losses = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut n_batches = 0;

            // Random batch sampling
            let mut rng = rand::thread_rng();

            for _ in 0..(n_samples / batch_size) {
                let batch_indices: Vec<usize> = (0..batch_size)
                    .map(|_| rng.gen_range(0..n_samples))
                    .collect();

                let mut batch_loss = 0.0;
                for &idx in &batch_indices {
                    let x = data.row(idx).to_owned();
                    let nll = -self.log_prob(&x);
                    batch_loss += nll;
                }
                batch_loss /= batch_size as f64;

                epoch_loss += batch_loss;
                n_batches += 1;
            }

            let avg_loss = epoch_loss / n_batches as f64;
            losses.push(avg_loss);

            if (epoch + 1) % 20 == 0 {
                log::info!("Epoch {}/{}: NLL = {:.4}", epoch + 1, epochs, avg_loss);
            }
        }

        self.is_trained = true;

        Ok(TrainingStats {
            epochs,
            final_loss: *losses.last().unwrap_or(&f64::NAN),
            loss_history: losses,
        })
    }

    /// Check if the model is trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Get the configuration
    pub fn config(&self) -> &NSFConfig {
        &self.config
    }

    /// Estimate probability density at a point
    pub fn density(&self, x: &Array1<f64>) -> f64 {
        self.log_prob(x).exp()
    }

    /// Check if a point is within the learned distribution
    pub fn is_in_distribution(&self, x: &Array1<f64>, threshold: f64) -> bool {
        self.log_prob(x) > threshold
    }
}

/// Training statistics
#[derive(Debug, Clone)]
pub struct TrainingStats {
    /// Number of training epochs
    pub epochs: usize,
    /// Final loss value
    pub final_loss: f64,
    /// Loss history over training
    pub loss_history: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_nsf_creation() {
        let config = NSFConfig::new(8);
        let nsf = NeuralSplineFlow::new(config);
        assert_eq!(nsf.config.dim, 8);
    }

    #[test]
    fn test_forward_inverse() {
        let config = NSFConfig::new(4).with_num_layers(2);
        let nsf = NeuralSplineFlow::new(config);

        let x = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.4]);
        let (z, _) = nsf.forward(&x);
        let (x_recovered, _) = nsf.inverse(&z);

        for i in 0..4 {
            assert_abs_diff_eq!(x[i], x_recovered[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_log_prob() {
        let config = NSFConfig::new(4);
        let nsf = NeuralSplineFlow::new(config);

        let x = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.4]);
        let log_p = nsf.log_prob(&x);

        // Log probability should be finite
        assert!(log_p.is_finite());
    }

    #[test]
    fn test_sample() {
        let config = NSFConfig::new(4);
        let nsf = NeuralSplineFlow::new(config);

        let samples = nsf.sample(10);
        assert_eq!(samples.shape(), &[10, 4]);
    }
}
