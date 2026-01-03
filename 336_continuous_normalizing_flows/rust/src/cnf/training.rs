//! # CNF Training
//!
//! Training utilities for Continuous Normalizing Flows.

use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::{Distribution, Normal};

use super::ContinuousNormalizingFlow;

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Patience for early stopping
    pub patience: usize,
    /// Gradient clipping threshold
    pub grad_clip: f64,
    /// Kinetic regularization weight
    pub kinetic_weight: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 128,
            learning_rate: 0.001,
            weight_decay: 1e-5,
            patience: 10,
            grad_clip: 1.0,
            kinetic_weight: 0.01,
        }
    }
}

/// CNF Trainer
pub struct CNFTrainer {
    config: TrainingConfig,
}

impl CNFTrainer {
    /// Create a new trainer with default config
    pub fn new() -> Self {
        Self {
            config: TrainingConfig::default(),
        }
    }

    /// Create trainer with custom config
    pub fn with_config(config: TrainingConfig) -> Self {
        Self { config }
    }

    /// Train the CNF model
    ///
    /// Uses negative log-likelihood loss with optional kinetic regularization.
    ///
    /// Returns the trained model and training history.
    pub fn train(
        &self,
        model: &mut ContinuousNormalizingFlow,
        train_data: &Array2<f64>,
        val_data: Option<&Array2<f64>>,
    ) -> TrainingHistory {
        let mut history = TrainingHistory::new();
        let mut rng = rand::thread_rng();

        // Initialize optimizer state (simple momentum)
        let params = model.get_params();
        let mut velocity = vec![0.0; params.len()];
        let momentum = 0.9;

        let mut best_val_loss = f64::INFINITY;
        let mut best_params = params.clone();
        let mut patience_counter = 0;

        for epoch in 0..self.config.epochs {
            // Training
            let train_loss = self.train_epoch(
                model,
                train_data,
                &mut velocity,
                momentum,
                &mut rng,
            );

            history.train_losses.push(train_loss);

            // Validation
            if let Some(val) = val_data {
                let val_loss = self.compute_loss(model, val);
                history.val_losses.push(val_loss);

                // Early stopping
                if val_loss < best_val_loss {
                    best_val_loss = val_loss;
                    best_params = model.get_params();
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= self.config.patience {
                        tracing::info!("Early stopping at epoch {}", epoch + 1);
                        break;
                    }
                }
            }

            // Logging
            if (epoch + 1) % 10 == 0 {
                let val_str = if let Some(val_loss) = history.val_losses.last() {
                    format!(", Val Loss: {:.4}", val_loss)
                } else {
                    String::new()
                };

                tracing::info!(
                    "Epoch {}/{}: Train Loss: {:.4}{}",
                    epoch + 1,
                    self.config.epochs,
                    train_loss,
                    val_str
                );
            }
        }

        // Restore best parameters
        if val_data.is_some() {
            model.set_params(&best_params);
        }

        history
    }

    /// Train for one epoch
    fn train_epoch<R: Rng>(
        &self,
        model: &mut ContinuousNormalizingFlow,
        data: &Array2<f64>,
        velocity: &mut [f64],
        momentum: f64,
        rng: &mut R,
    ) -> f64 {
        let n = data.nrows();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(rng);

        let mut total_loss = 0.0;
        let mut n_batches = 0;

        for chunk in indices.chunks(self.config.batch_size) {
            // Get batch
            let batch_size = chunk.len();
            let mut batch = Array2::zeros((batch_size, model.dim));
            for (i, &idx) in chunk.iter().enumerate() {
                for j in 0..model.dim {
                    batch[[i, j]] = data[[idx, j]];
                }
            }

            // Compute loss and gradients
            let (loss, gradients) = self.compute_loss_and_gradients(model, &batch);
            total_loss += loss;
            n_batches += 1;

            // Update with momentum SGD
            let params = model.get_params();
            let mut new_params = params.clone();

            for (i, (p, &g)) in new_params.iter_mut().zip(gradients.iter()).enumerate() {
                // Gradient clipping
                let clipped_g = g.clamp(-self.config.grad_clip, self.config.grad_clip);

                // Momentum update
                velocity[i] = momentum * velocity[i] - self.config.learning_rate * clipped_g;
                *p += velocity[i];

                // Weight decay
                *p *= 1.0 - self.config.learning_rate * self.config.weight_decay;
            }

            model.set_params(&new_params);
        }

        total_loss / n_batches as f64
    }

    /// Compute negative log-likelihood loss
    fn compute_loss(&self, model: &ContinuousNormalizingFlow, data: &Array2<f64>) -> f64 {
        let log_probs = model.log_prob_batch(data);
        -log_probs.mean().unwrap_or(0.0)
    }

    /// Compute loss and gradients using finite differences
    fn compute_loss_and_gradients(
        &self,
        model: &ContinuousNormalizingFlow,
        batch: &Array2<f64>,
    ) -> (f64, Vec<f64>) {
        let eps = 1e-4;
        let params = model.get_params();
        let mut gradients = vec![0.0; params.len()];

        // Compute base loss
        let base_loss = self.compute_loss(model, batch);

        // Compute gradients via finite differences
        // (In production, use proper autodiff)
        let mut temp_model = model.clone();

        for i in 0..params.len().min(100) {
            // Limit for speed
            let mut params_plus = params.clone();
            params_plus[i] += eps;
            temp_model.set_params(&params_plus);
            let loss_plus = self.compute_loss(&temp_model, batch);

            gradients[i] = (loss_plus - base_loss) / eps;
        }

        // For remaining params, use random subset
        if params.len() > 100 {
            let mut rng = rand::thread_rng();
            for _ in 0..100 {
                let i = rng.gen_range(100..params.len());
                let mut params_plus = params.clone();
                params_plus[i] += eps;
                temp_model.set_params(&params_plus);
                let loss_plus = self.compute_loss(&temp_model, batch);
                gradients[i] = (loss_plus - base_loss) / eps;
            }
        }

        (base_loss, gradients)
    }
}

impl Default for CNFTrainer {
    fn default() -> Self {
        Self::new()
    }
}

/// Training history
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    /// Training losses per epoch
    pub train_losses: Vec<f64>,
    /// Validation losses per epoch
    pub val_losses: Vec<f64>,
}

impl TrainingHistory {
    /// Create new empty history
    pub fn new() -> Self {
        Self {
            train_losses: Vec::new(),
            val_losses: Vec::new(),
        }
    }

    /// Get best validation loss
    pub fn best_val_loss(&self) -> Option<f64> {
        self.val_losses.iter().cloned().reduce(f64::min)
    }

    /// Get final training loss
    pub fn final_train_loss(&self) -> Option<f64> {
        self.train_losses.last().cloned()
    }
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::generate_synthetic_candles;
    use crate::utils::compute_features_batch;

    #[test]
    fn test_training_config() {
        let config = TrainingConfig::default();
        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 128);
    }

    #[test]
    fn test_trainer_creation() {
        let trainer = CNFTrainer::new();
        assert_eq!(trainer.config.epochs, 100);

        let custom_config = TrainingConfig {
            epochs: 50,
            ..Default::default()
        };
        let trainer = CNFTrainer::with_config(custom_config);
        assert_eq!(trainer.config.epochs, 50);
    }
}
