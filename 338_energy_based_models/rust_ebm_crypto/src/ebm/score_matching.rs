//! Score Matching training for Energy-Based Models
//!
//! Score matching provides an alternative to contrastive divergence
//! that doesn't require MCMC sampling. It trains the model to match
//! the score (gradient of log-density) of the data distribution.

use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::Normal;

use super::energy_net::{EnergyModel, Layer, Activation};

/// Score Matching trainer for EBM
///
/// Implements Denoising Score Matching (DSM) which is more stable
/// than vanilla score matching.
#[derive(Debug, Clone)]
pub struct ScoreMatchingTrainer {
    /// Noise scale for denoising score matching
    pub noise_scale: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of noise samples per data point
    pub n_noise_samples: usize,
}

impl Default for ScoreMatchingTrainer {
    fn default() -> Self {
        Self {
            noise_scale: 0.1,
            learning_rate: 0.001,
            n_noise_samples: 1,
        }
    }
}

impl ScoreMatchingTrainer {
    /// Create a new score matching trainer
    pub fn new(noise_scale: f64, learning_rate: f64) -> Self {
        Self {
            noise_scale,
            learning_rate,
            n_noise_samples: 1,
        }
    }

    /// Train the energy model using denoising score matching
    ///
    /// The idea: Add noise to data, then train the model to predict
    /// the score (gradient pointing back to clean data).
    pub fn train(&self, model: &mut EnergyModel, data: &Array2<f64>, epochs: usize) {
        let n_samples = data.nrows();
        let input_dim = data.ncols();
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        log::info!(
            "Training with Denoising Score Matching for {} epochs",
            epochs
        );

        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for i in 0..n_samples {
                let x = data.row(i).to_owned();

                // Add noise
                let noise: Array1<f64> =
                    Array1::from_shape_fn(input_dim, |_| rng.sample(normal) * self.noise_scale);
                let x_noisy = &x + &noise;

                // Compute model score (gradient of energy w.r.t. input)
                let model_score = self.compute_score(model, &x_noisy);

                // Target score: gradient of log p_noise(x_noisy | x) = -noise / sigma^2
                let target_score = -&noise / (self.noise_scale * self.noise_scale);

                // Score matching loss: ||model_score - target_score||^2
                let diff = &model_score - &target_score;
                let loss: f64 = diff.iter().map(|d| d * d).sum();
                total_loss += loss;

                // Update model (simplified gradient descent)
                self.update_model(model, &x_noisy, &target_score);
            }

            if epoch % 10 == 0 || epoch == epochs - 1 {
                log::info!(
                    "Epoch {}/{}: score_matching_loss = {:.6}",
                    epoch + 1,
                    epochs,
                    total_loss / n_samples as f64
                );
            }
        }
    }

    /// Compute the score (gradient of energy) at a point
    fn compute_score(&self, model: &EnergyModel, x: &Array1<f64>) -> Array1<f64> {
        let eps = 1e-5;
        let input_dim = x.len();
        let mut score = Array1::zeros(input_dim);

        for i in 0..input_dim {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += eps;
            x_minus[i] -= eps;

            // Score is negative gradient of energy
            score[i] = -(model.energy(&x_plus) - model.energy(&x_minus)) / (2.0 * eps);
        }

        score
    }

    /// Update model weights to reduce score matching loss
    fn update_model(&self, model: &mut EnergyModel, x: &Array1<f64>, target_score: &Array1<f64>) {
        let eps = 1e-4;
        let current_score = self.compute_score(model, x);
        let score_error: f64 = (&current_score - target_score)
            .iter()
            .map(|d| d * d)
            .sum();

        // Update each layer's weights
        for layer in &mut model.layers {
            for i in 0..layer.weights.nrows() {
                for j in 0..layer.weights.ncols() {
                    let original = layer.weights[[i, j]];

                    layer.weights[[i, j]] = original + eps;
                    let score_plus = self.compute_score(model, x);
                    let error_plus: f64 = (&score_plus - target_score)
                        .iter()
                        .map(|d| d * d)
                        .sum();

                    layer.weights[[i, j]] = original;

                    let grad = (error_plus - score_error) / eps;
                    layer.weights[[i, j]] -= self.learning_rate * grad;
                }
            }

            // Update biases
            for j in 0..layer.bias.len() {
                let original = layer.bias[j];

                layer.bias[j] = original + eps;
                let score_plus = self.compute_score(model, x);
                let error_plus: f64 = (&score_plus - target_score)
                    .iter()
                    .map(|d| d * d)
                    .sum();

                layer.bias[j] = original;

                let grad = (error_plus - score_error) / eps;
                layer.bias[j] -= self.learning_rate * grad;
            }
        }
    }
}

/// Sliced Score Matching (more efficient for high dimensions)
///
/// Instead of matching the full score, we match the score projected
/// onto random directions. This scales better with dimensionality.
#[derive(Debug, Clone)]
pub struct SlicedScoreMatching {
    /// Number of random projections
    pub n_projections: usize,
    /// Noise scale
    pub noise_scale: f64,
    /// Learning rate
    pub learning_rate: f64,
}

impl Default for SlicedScoreMatching {
    fn default() -> Self {
        Self {
            n_projections: 4,
            noise_scale: 0.1,
            learning_rate: 0.001,
        }
    }
}

impl SlicedScoreMatching {
    /// Train using sliced score matching
    pub fn train(&self, model: &mut EnergyModel, data: &Array2<f64>, epochs: usize) {
        let n_samples = data.nrows();
        let input_dim = data.ncols();
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        log::info!(
            "Training with Sliced Score Matching ({} projections) for {} epochs",
            self.n_projections,
            epochs
        );

        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for i in 0..n_samples {
                let x = data.row(i).to_owned();

                for _ in 0..self.n_projections {
                    // Random projection direction
                    let mut v: Array1<f64> =
                        Array1::from_shape_fn(input_dim, |_| rng.sample(normal));

                    // Normalize to unit vector
                    let norm = (v.iter().map(|x| x * x).sum::<f64>()).sqrt();
                    v.mapv_inplace(|x| x / norm);

                    // Add noise in projection direction
                    let noise_magnitude: f64 = rng.sample(normal) * self.noise_scale;
                    let x_noisy = &x + noise_magnitude * &v;

                    // Compute directional derivative of energy
                    let eps = 1e-5;
                    let e_plus = model.energy(&(&x_noisy + eps * &v));
                    let e_minus = model.energy(&(&x_noisy - eps * &v));
                    let directional_score = -(e_plus - e_minus) / (2.0 * eps);

                    // Target directional score
                    let target = -noise_magnitude / (self.noise_scale * self.noise_scale);

                    // Loss
                    let loss = (directional_score - target).powi(2);
                    total_loss += loss;
                }
            }

            if epoch % 10 == 0 || epoch == epochs - 1 {
                log::info!(
                    "Epoch {}/{}: sliced_score_loss = {:.6}",
                    epoch + 1,
                    epochs,
                    total_loss / (n_samples * self.n_projections) as f64
                );
            }
        }
    }
}

/// Kernel-based energy estimator
///
/// Uses kernel density estimation to estimate energy without
/// explicit parametric model. Useful as a baseline or for
/// comparison.
#[derive(Debug, Clone)]
pub struct KernelEnergyEstimator {
    /// Reference data points
    pub data: Array2<f64>,
    /// Kernel bandwidth
    pub bandwidth: f64,
}

impl KernelEnergyEstimator {
    /// Create a new kernel energy estimator
    pub fn new(data: Array2<f64>, bandwidth: f64) -> Self {
        Self { data, bandwidth }
    }

    /// Automatically select bandwidth using Scott's rule
    pub fn with_auto_bandwidth(data: Array2<f64>) -> Self {
        let n = data.nrows() as f64;
        let d = data.ncols() as f64;

        // Scott's rule: h = n^(-1/(d+4))
        let bandwidth = n.powf(-1.0 / (d + 4.0));

        Self { data, bandwidth }
    }

    /// Compute energy (negative log-density) at a point
    pub fn energy(&self, x: &Array1<f64>) -> f64 {
        let n = self.data.nrows() as f64;
        let d = self.data.ncols() as f64;

        // Kernel density estimate
        let mut density = 0.0;

        for i in 0..self.data.nrows() {
            let xi = self.data.row(i);
            let diff: f64 = x.iter().zip(xi.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            let kernel_val = (-diff / (2.0 * self.bandwidth * self.bandwidth)).exp();
            density += kernel_val;
        }

        // Normalize
        density /= n * (2.0 * std::f64::consts::PI * self.bandwidth * self.bandwidth).powf(d / 2.0);

        // Energy = -log(density)
        if density > 1e-300 {
            -density.ln()
        } else {
            700.0 // Max energy (avoid infinity)
        }
    }

    /// Compute energy for a batch
    pub fn energy_batch(&self, data: &Array2<f64>) -> Array1<f64> {
        let n = data.nrows();
        let mut energies = Array1::zeros(n);

        for i in 0..n {
            energies[i] = self.energy(&data.row(i).to_owned());
        }

        energies
    }

    /// Compute anomaly scores
    pub fn anomaly_scores(&self, data: &Array2<f64>) -> Array1<f64> {
        let energies = self.energy_batch(data);

        // Normalize to [0, 1]
        let min = energies.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        if range < 1e-10 {
            Array1::zeros(energies.len())
        } else {
            energies.mapv(|e| (e - min) / range)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_score_matching_trainer() {
        let trainer = ScoreMatchingTrainer::default();
        assert!(trainer.noise_scale > 0.0);
        assert!(trainer.learning_rate > 0.0);
    }

    #[test]
    fn test_kernel_estimator() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let estimator = KernelEnergyEstimator::with_auto_bandwidth(data);

        let x = array![2.0, 3.0]; // Close to training data
        let y = array![10.0, 10.0]; // Far from training data

        let energy_x = estimator.energy(&x);
        let energy_y = estimator.energy(&y);

        // Point far from data should have higher energy
        assert!(energy_y > energy_x);
    }
}
