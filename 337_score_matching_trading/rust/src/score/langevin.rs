//! # Langevin Dynamics
//!
//! Implementation of annealed Langevin dynamics for sampling from learned distributions.
//!
//! Langevin dynamics uses the score function to generate samples:
//! x_{t+1} = x_t + α∇_x log p(x_t) + √(2α)ε, where ε ~ N(0, I)

use super::network::MultiScaleScoreNetwork;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};

/// Configuration for Langevin dynamics
#[derive(Debug, Clone)]
pub struct LangevinConfig {
    /// Number of steps per noise level
    pub n_steps_per_level: usize,
    /// Base step size
    pub step_size: f64,
    /// Whether to use annealing
    pub use_annealing: bool,
    /// Final noise factor for sampling
    pub final_noise: f64,
}

impl Default for LangevinConfig {
    fn default() -> Self {
        Self {
            n_steps_per_level: 100,
            step_size: 0.01,
            use_annealing: true,
            final_noise: 0.0,
        }
    }
}

/// Langevin Dynamics sampler
///
/// Uses the score function to:
/// 1. Generate samples from the learned distribution
/// 2. Denoise noisy inputs
/// 3. Predict likely movements from current state
pub struct LangevinDynamics<'a> {
    /// Reference to score network
    network: &'a MultiScaleScoreNetwork,
    /// Configuration
    config: LangevinConfig,
}

impl<'a> LangevinDynamics<'a> {
    /// Create new Langevin dynamics sampler
    pub fn new(network: &'a MultiScaleScoreNetwork, config: LangevinConfig) -> Self {
        Self { network, config }
    }

    /// Generate samples using annealed Langevin dynamics
    ///
    /// Starts from random noise and iteratively refines using score function
    /// across all noise levels (from high to low).
    ///
    /// # Arguments
    ///
    /// * `n_samples` - Number of samples to generate
    ///
    /// # Returns
    ///
    /// Generated samples (n_samples x input_dim)
    pub fn sample(&self, n_samples: usize) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let dim = self.network.input_dim;

        // Initialize from random noise
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut samples = Array2::from_shape_fn((n_samples, dim), |_| normal.sample(&mut rng));

        if self.config.use_annealing {
            // Anneal through noise levels
            for level in 0..self.network.num_noise_levels {
                let sigma = self.network.get_sigma(level);
                let alpha = self.config.step_size * sigma * sigma;

                for _ in 0..self.config.n_steps_per_level {
                    for i in 0..n_samples {
                        let x = samples.row(i).to_owned();
                        let score = self.network.forward(&x, level);

                        // Langevin update
                        let noise: Array1<f64> = Array1::from_iter(
                            (0..dim).map(|_| normal.sample(&mut rng))
                        );

                        let new_x = &x + &(&score * alpha) + &(&noise * (2.0 * alpha).sqrt());
                        samples.row_mut(i).assign(&new_x);
                    }
                }
            }
        } else {
            // Single noise level (lowest)
            let level = self.network.num_noise_levels - 1;
            let sigma = self.network.get_sigma(level);
            let alpha = self.config.step_size * sigma * sigma;

            for _ in 0..(self.config.n_steps_per_level * self.network.num_noise_levels) {
                for i in 0..n_samples {
                    let x = samples.row(i).to_owned();
                    let score = self.network.forward(&x, level);

                    let noise: Array1<f64> = Array1::from_iter(
                        (0..dim).map(|_| normal.sample(&mut rng))
                    );

                    let new_x = &x + &(&score * alpha) + &(&noise * (2.0 * alpha).sqrt());
                    samples.row_mut(i).assign(&new_x);
                }
            }
        }

        samples
    }

    /// Denoise input by following score at lowest noise level
    ///
    /// # Arguments
    ///
    /// * `x` - Noisy input to denoise
    /// * `n_steps` - Number of denoising steps
    ///
    /// # Returns
    ///
    /// Denoised output
    pub fn denoise(&self, x: &Array1<f64>, n_steps: usize) -> Array1<f64> {
        let level = self.network.num_noise_levels - 1;
        let sigma = self.network.get_sigma(level);
        let alpha = self.config.step_size * sigma * sigma;

        let mut result = x.clone();

        for _ in 0..n_steps {
            let score = self.network.forward(&result, level);
            result = &result + &(&score * alpha);
        }

        result
    }

    /// Denoise batch of inputs
    pub fn denoise_batch(&self, x: &Array2<f64>, n_steps: usize) -> Array2<f64> {
        let batch_size = x.nrows();
        let mut result = Array2::zeros(x.dim());

        for i in 0..batch_size {
            let denoised = self.denoise(&x.row(i).to_owned(), n_steps);
            result.row_mut(i).assign(&denoised);
        }

        result
    }

    /// Get trading signal from score function
    ///
    /// Uses the score at the return feature index to predict direction.
    ///
    /// # Arguments
    ///
    /// * `x` - Current market state
    /// * `return_idx` - Index of return feature in state vector
    ///
    /// # Returns
    ///
    /// (signal, confidence) tuple where:
    /// - signal: +1 for long, -1 for short, 0 for neutral
    /// - confidence: magnitude of signal (0 to 1)
    pub fn get_trading_signal(&self, x: &Array1<f64>, return_idx: usize) -> (f64, f64) {
        let level = self.network.num_noise_levels - 1;
        let score = self.network.forward(x, level);

        let return_score = score[return_idx];
        let signal = return_score.signum();
        let confidence = return_score.abs();

        // Normalize confidence to 0-1 range (using sigmoid-like scaling)
        let normalized_confidence = confidence / (1.0 + confidence);

        (signal, normalized_confidence)
    }

    /// Estimate local probability density (up to normalization constant)
    ///
    /// Uses path integral of score function:
    /// log p(x) - log p(x_0) = ∫ s(x(t)) · dx(t)
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to estimate density
    /// * `n_steps` - Number of integration steps
    ///
    /// # Returns
    ///
    /// Log density estimate (relative to origin)
    pub fn estimate_log_density(&self, x: &Array1<f64>, n_steps: usize) -> f64 {
        let dim = x.len();
        let level = self.network.num_noise_levels - 1;

        // Reference point (origin)
        let x_0 = Array1::zeros(dim);

        let mut log_density = 0.0;

        for i in 0..n_steps {
            let t = i as f64 / n_steps as f64;
            let x_t = &x_0 * (1.0 - t) + &(x * t);

            let score = self.network.forward(&x_t, level);

            // Path direction
            let dx = x / n_steps as f64;

            // Dot product
            let contribution: f64 = score.iter().zip(dx.iter()).map(|(s, d)| s * d).sum();
            log_density += contribution;
        }

        log_density
    }

    /// Check if point is within learned distribution
    ///
    /// # Arguments
    ///
    /// * `x` - Point to check
    /// * `threshold` - Log density threshold (points below this are out of distribution)
    ///
    /// # Returns
    ///
    /// True if point is in distribution
    pub fn is_in_distribution(&self, x: &Array1<f64>, threshold: f64) -> bool {
        let log_density = self.estimate_log_density(x, 50);
        log_density > threshold
    }

    /// Predict future state by following score
    ///
    /// This can be used to estimate where the market state is heading.
    ///
    /// # Arguments
    ///
    /// * `x` - Current state
    /// * `n_steps` - Number of prediction steps
    /// * `step_scale` - Scale factor for step size
    ///
    /// # Returns
    ///
    /// Predicted future state
    pub fn predict_movement(&self, x: &Array1<f64>, n_steps: usize, step_scale: f64) -> Array1<f64> {
        let level = self.network.num_noise_levels - 1;
        let sigma = self.network.get_sigma(level);
        let alpha = self.config.step_size * sigma * sigma * step_scale;

        let mut result = x.clone();

        for _ in 0..n_steps {
            let score = self.network.forward(&result, level);
            result = &result + &(&score * alpha);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_langevin_config_default() {
        let config = LangevinConfig::default();
        assert_eq!(config.n_steps_per_level, 100);
        assert_eq!(config.step_size, 0.01);
        assert!(config.use_annealing);
    }

    #[test]
    fn test_denoise() {
        let network = MultiScaleScoreNetwork::new(5, 16, 2, 5, 0.01, 1.0);
        let config = LangevinConfig::default();
        let langevin = LangevinDynamics::new(&network, config);

        let noisy = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let denoised = langevin.denoise(&noisy, 10);

        assert_eq!(denoised.len(), 5);
    }

    #[test]
    fn test_trading_signal() {
        let network = MultiScaleScoreNetwork::new(5, 16, 2, 5, 0.01, 1.0);
        let config = LangevinConfig::default();
        let langevin = LangevinDynamics::new(&network, config);

        let state = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        let (signal, confidence) = langevin.get_trading_signal(&state, 0);

        assert!(signal.abs() <= 1.0);
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_in_distribution() {
        let network = MultiScaleScoreNetwork::new(5, 16, 2, 5, 0.01, 1.0);
        let config = LangevinConfig::default();
        let langevin = LangevinDynamics::new(&network, config);

        let point = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]);

        // Just check it runs without errors
        let _ = langevin.is_in_distribution(&point, -10.0);
    }
}
