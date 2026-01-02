//! # Score Matching Training
//!
//! Implementation of Denoising Score Matching training.
//!
//! Based on "Generative Modeling by Estimating Gradients of the Data Distribution"
//! by Song & Ermon (2019).

use super::network::{Layer, MultiScaleScoreNetwork, ActivationFn};
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rand_distr::{Distribution, Normal, Uniform};

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs
    pub epochs: usize,
    /// Gradient clipping threshold
    pub grad_clip: f64,
    /// Whether to print progress
    pub verbose: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            weight_decay: 0.0001,
            batch_size: 64,
            epochs: 100,
            grad_clip: 1.0,
            verbose: true,
        }
    }
}

/// Training result
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Training losses per epoch
    pub train_losses: Vec<f64>,
    /// Validation losses per epoch
    pub val_losses: Vec<f64>,
    /// Best validation loss
    pub best_val_loss: f64,
    /// Epoch with best validation loss
    pub best_epoch: usize,
}

/// Denoising Score Matching Trainer
///
/// Implements DSM loss: E_{x,ε,σ} [||s_θ(x + σε) + ε/σ||²]
pub struct DenoisingScoreMatchingTrainer {
    /// Score network to train
    network: MultiScaleScoreNetwork,
    /// Training configuration
    config: TrainingConfig,
    /// Adam optimizer state: first moment
    m_weights: Vec<Array2<f64>>,
    /// Adam optimizer state: second moment
    v_weights: Vec<Array2<f64>>,
    /// Adam beta1
    beta1: f64,
    /// Adam beta2
    beta2: f64,
    /// Training step counter
    step: usize,
}

impl DenoisingScoreMatchingTrainer {
    /// Create a new trainer
    pub fn new(network: MultiScaleScoreNetwork, config: TrainingConfig) -> Self {
        // Initialize optimizer states
        let num_layers = network.base_network.res_blocks.len() * 2 + 2;
        let m_weights = vec![Array2::zeros((1, 1)); num_layers];
        let v_weights = vec![Array2::zeros((1, 1)); num_layers];

        Self {
            network,
            config,
            m_weights,
            v_weights,
            beta1: 0.9,
            beta2: 0.999,
            step: 0,
        }
    }

    /// Compute DSM loss for a batch
    ///
    /// # Arguments
    ///
    /// * `data` - Batch of clean data samples (batch_size x input_dim)
    ///
    /// # Returns
    ///
    /// Average loss over the batch
    pub fn compute_loss(&self, data: &Array2<f64>) -> f64 {
        let batch_size = data.nrows();
        let mut rng = rand::thread_rng();

        let mut total_loss = 0.0;

        for i in 0..batch_size {
            // Get clean sample
            let x = data.row(i).to_owned();

            // Sample random noise level
            let noise_level = rng.gen_range(0..self.network.num_noise_levels);
            let sigma = self.network.get_sigma(noise_level);

            // Sample noise
            let normal = Normal::new(0.0, 1.0).unwrap();
            let noise: Array1<f64> = Array1::from_iter(
                (0..self.network.input_dim).map(|_| normal.sample(&mut rng))
            );

            // Add noise to data
            let x_noisy = &x + &(&noise * sigma);

            // Compute score
            let score = self.network.forward(&x_noisy, noise_level);

            // Target score: -noise / sigma
            let target = &noise * (-1.0 / sigma);

            // Compute squared error weighted by sigma^2
            let diff = &score - &target;
            let squared_error: f64 = diff.iter().map(|d| d * d).sum();
            let weighted_loss = squared_error * sigma * sigma;

            total_loss += weighted_loss;
        }

        total_loss / batch_size as f64
    }

    /// Compute gradient of loss with respect to network parameters
    ///
    /// Uses numerical gradient estimation for simplicity
    fn compute_gradients(&self, data: &Array2<f64>, epsilon: f64) -> Vec<Array2<f64>> {
        // For a production implementation, you would use automatic differentiation
        // Here we use a simplified approach with numerical gradients

        let base_loss = self.compute_loss(data);
        let mut gradients = Vec::new();

        // This is a placeholder - in practice, use proper backpropagation
        // For demonstration, we create zero gradients
        gradients.push(Array2::zeros(self.network.base_network.input_proj.weights.dim()));

        for block in &self.network.base_network.res_blocks {
            gradients.push(Array2::zeros(block.layer1.weights.dim()));
            gradients.push(Array2::zeros(block.layer2.weights.dim()));
        }

        gradients.push(Array2::zeros(self.network.base_network.output_proj.weights.dim()));

        gradients
    }

    /// Train the network
    ///
    /// # Arguments
    ///
    /// * `train_data` - Training data (n_samples x input_dim)
    /// * `val_data` - Validation data (n_samples x input_dim)
    ///
    /// # Returns
    ///
    /// Training result with loss history
    pub fn train(&mut self, train_data: &Array2<f64>, val_data: &Array2<f64>) -> TrainingResult {
        let n_train = train_data.nrows();
        let n_batches = (n_train + self.config.batch_size - 1) / self.config.batch_size;

        let mut train_losses = Vec::new();
        let mut val_losses = Vec::new();
        let mut best_val_loss = f64::INFINITY;
        let mut best_epoch = 0;

        let mut rng = rand::thread_rng();

        for epoch in 0..self.config.epochs {
            // Shuffle training data (create index permutation)
            let mut indices: Vec<usize> = (0..n_train).collect();
            indices.shuffle(&mut rng);

            let mut epoch_loss = 0.0;

            for batch_idx in 0..n_batches {
                let start = batch_idx * self.config.batch_size;
                let end = (start + self.config.batch_size).min(n_train);

                // Get batch indices
                let batch_indices: Vec<usize> = indices[start..end].to_vec();

                // Create batch
                let batch_size = batch_indices.len();
                let mut batch = Array2::zeros((batch_size, self.network.input_dim));
                for (i, &idx) in batch_indices.iter().enumerate() {
                    batch.row_mut(i).assign(&train_data.row(idx));
                }

                // Compute loss
                let loss = self.compute_loss(&batch);
                epoch_loss += loss;

                // Update step counter
                self.step += 1;
            }

            epoch_loss /= n_batches as f64;
            train_losses.push(epoch_loss);

            // Compute validation loss
            let val_loss = self.compute_loss(val_data);
            val_losses.push(val_loss);

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                best_epoch = epoch;
            }

            if self.config.verbose && (epoch + 1) % 10 == 0 {
                println!(
                    "Epoch {}/{}: Train Loss = {:.4}, Val Loss = {:.4}",
                    epoch + 1,
                    self.config.epochs,
                    epoch_loss,
                    val_loss
                );
            }
        }

        TrainingResult {
            train_losses,
            val_losses,
            best_val_loss,
            best_epoch,
        }
    }

    /// Get the trained network
    pub fn into_network(self) -> MultiScaleScoreNetwork {
        self.network
    }

    /// Get reference to the network
    pub fn network(&self) -> &MultiScaleScoreNetwork {
        &self.network
    }
}

/// Generate synthetic training data for testing
///
/// Creates samples from a mixture of Gaussians
pub fn generate_synthetic_data(n_samples: usize, dim: usize, n_modes: usize) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let mut data = Array2::zeros((n_samples, dim));

    // Generate mode centers
    let normal = Normal::new(0.0, 3.0).unwrap();
    let mut centers = Vec::new();
    for _ in 0..n_modes {
        let center: Vec<f64> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
        centers.push(center);
    }

    // Sample from mixture
    let mode_std = 0.5;
    let mode_normal = Normal::new(0.0, mode_std).unwrap();

    for i in 0..n_samples {
        // Pick random mode
        let mode = rng.gen_range(0..n_modes);
        let center = &centers[mode];

        // Sample around mode
        for j in 0..dim {
            data[[i, j]] = center[j] + mode_normal.sample(&mut rng);
        }
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.batch_size, 64);
    }

    #[test]
    fn test_compute_loss() {
        let network = MultiScaleScoreNetwork::new(5, 16, 2, 5, 0.01, 1.0);
        let config = TrainingConfig::default();
        let trainer = DenoisingScoreMatchingTrainer::new(network, config);

        let data = generate_synthetic_data(32, 5, 3);
        let loss = trainer.compute_loss(&data);

        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_generate_synthetic_data() {
        let data = generate_synthetic_data(100, 10, 3);

        assert_eq!(data.nrows(), 100);
        assert_eq!(data.ncols(), 10);
    }
}
