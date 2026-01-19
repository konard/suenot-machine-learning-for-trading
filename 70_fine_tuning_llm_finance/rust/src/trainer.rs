//! Training pipeline for fine-tuning models.
//!
//! This module provides:
//! - `TrainingConfig`: Configuration for training
//! - `TrainingMetrics`: Metrics tracked during training
//! - `FineTuningTrainer`: Main training loop with early stopping

use crate::model::FinancialSentimentModel;
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Batch size
    pub batch_size: usize,
    /// Number of training epochs
    pub num_epochs: usize,
    /// Early stopping patience (epochs)
    pub early_stopping_patience: usize,
    /// Weight decay for regularization
    pub weight_decay: f32,
    /// Number of warmup steps for learning rate schedule
    pub warmup_steps: usize,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f32,
    /// Whether to use gradient checkpointing
    pub gradient_checkpointing: bool,
    /// Random seed
    pub seed: u64,
    /// Logging interval (steps)
    pub log_interval: usize,
    /// Evaluation interval (steps)
    pub eval_interval: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            batch_size: 32,
            num_epochs: 10,
            early_stopping_patience: 5,
            weight_decay: 0.01,
            warmup_steps: 100,
            max_grad_norm: 1.0,
            gradient_checkpointing: false,
            seed: 42,
            log_interval: 100,
            eval_interval: 500,
        }
    }
}

impl TrainingConfig {
    /// Create a new training configuration with custom learning rate.
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set number of epochs.
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.num_epochs = epochs;
        self
    }

    /// Set early stopping patience.
    pub fn with_patience(mut self, patience: usize) -> Self {
        self.early_stopping_patience = patience;
        self
    }
}

/// Metrics tracked during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Training loss history
    pub train_losses: Vec<f32>,
    /// Validation loss history
    pub val_losses: Vec<f32>,
    /// Training accuracy history
    pub train_accuracies: Vec<f32>,
    /// Validation accuracy history
    pub val_accuracies: Vec<f32>,
    /// Best validation loss achieved
    pub best_val_loss: f32,
    /// Epoch at which best validation loss was achieved
    pub best_epoch: usize,
    /// Total training time in seconds
    pub training_time_secs: f64,
    /// Number of early stopping triggers
    pub early_stop_count: usize,
    /// Whether training was stopped early
    pub stopped_early: bool,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self {
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            train_accuracies: Vec::new(),
            val_accuracies: Vec::new(),
            best_val_loss: f32::INFINITY,
            best_epoch: 0,
            training_time_secs: 0.0,
            early_stop_count: 0,
            stopped_early: false,
        }
    }
}

/// Fine-tuning trainer with early stopping.
pub struct FineTuningTrainer {
    /// Model to train
    pub model: FinancialSentimentModel,
    /// Training configuration
    pub config: TrainingConfig,
    /// Training metrics
    pub metrics: TrainingMetrics,
    /// Current learning rate
    current_lr: f32,
    /// Global step counter
    global_step: usize,
}

impl FineTuningTrainer {
    /// Create a new trainer.
    pub fn new(model: FinancialSentimentModel, config: TrainingConfig) -> Self {
        let current_lr = config.learning_rate;
        Self {
            model,
            config,
            metrics: TrainingMetrics::default(),
            current_lr,
            global_step: 0,
        }
    }

    /// Get learning rate with warmup schedule.
    fn get_learning_rate(&self) -> f32 {
        if self.global_step < self.config.warmup_steps {
            // Linear warmup
            self.config.learning_rate * (self.global_step as f32 / self.config.warmup_steps as f32)
        } else {
            self.config.learning_rate
        }
    }

    /// Compute cross-entropy loss.
    fn cross_entropy_loss(logits: &Array1<f32>, target: usize) -> f32 {
        // Softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Array1<f32> = logits.mapv(|x| (x - max_logit).exp());
        let sum_exp = exp_logits.sum();
        let log_softmax = logits.mapv(|x| x - max_logit - sum_exp.ln());

        // Negative log likelihood
        -log_softmax[target]
    }

    /// Compute accuracy.
    fn compute_accuracy(predictions: &[usize], targets: &[usize]) -> f32 {
        let correct = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(p, t)| p == t)
            .count();
        correct as f32 / predictions.len() as f32
    }

    /// Train for one epoch.
    fn train_epoch(
        &mut self,
        train_x: &Array2<f32>,
        train_y: &[usize],
    ) -> (f32, f32) {
        let n_samples = train_x.nrows();
        let batch_size = self.config.batch_size.min(n_samples);
        let n_batches = (n_samples + batch_size - 1) / batch_size;

        let mut total_loss = 0.0;
        let mut all_predictions = Vec::new();

        // Simple random permutation for shuffling
        let mut indices: Vec<usize> = (0..n_samples).collect();
        // Fisher-Yates shuffle
        for i in (1..n_samples).rev() {
            let j = (self.global_step * 1337 + i) % (i + 1);
            indices.swap(i, j);
        }

        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(n_samples);

            let mut batch_loss = 0.0;

            for &idx in &indices[start..end] {
                let input = train_x.row(idx).to_owned();
                let target = train_y[idx];

                // Forward pass
                let logits = self.model.forward(&input);
                let loss = Self::cross_entropy_loss(&logits, target);
                batch_loss += loss;

                // Get prediction
                let pred = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                all_predictions.push(pred);

                self.global_step += 1;
            }

            total_loss += batch_loss;
        }

        let avg_loss = total_loss / n_samples as f32;
        let accuracy = Self::compute_accuracy(&all_predictions, train_y);

        (avg_loss, accuracy)
    }

    /// Evaluate on validation data.
    fn evaluate(&self, val_x: &Array2<f32>, val_y: &[usize]) -> (f32, f32) {
        let n_samples = val_x.nrows();
        let mut total_loss = 0.0;
        let mut predictions = Vec::new();

        for i in 0..n_samples {
            let input = val_x.row(i).to_owned();
            let target = val_y[i];

            let logits = self.model.forward(&input);
            let loss = Self::cross_entropy_loss(&logits, target);
            total_loss += loss;

            let pred = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            predictions.push(pred);
        }

        let avg_loss = total_loss / n_samples as f32;
        let accuracy = Self::compute_accuracy(&predictions, val_y);

        (avg_loss, accuracy)
    }

    /// Run the full training loop.
    ///
    /// # Arguments
    ///
    /// * `train_x` - Training features of shape (n_train, dim)
    /// * `train_y` - Training labels
    /// * `val_x` - Validation features of shape (n_val, dim)
    /// * `val_y` - Validation labels
    ///
    /// # Returns
    ///
    /// Training metrics after completion.
    pub fn train(
        &mut self,
        train_x: &Array2<f32>,
        train_y: &[usize],
        val_x: &Array2<f32>,
        val_y: &[usize],
    ) -> TrainingMetrics {
        let start_time = Instant::now();
        let mut patience_counter = 0;

        println!("Starting training...");
        println!("  Train samples: {}", train_x.nrows());
        println!("  Val samples: {}", val_x.nrows());
        println!("  Epochs: {}", self.config.num_epochs);
        println!("  Batch size: {}", self.config.batch_size);
        println!();

        for epoch in 0..self.config.num_epochs {
            // Training
            let (train_loss, train_acc) = self.train_epoch(train_x, train_y);
            self.metrics.train_losses.push(train_loss);
            self.metrics.train_accuracies.push(train_acc);

            // Validation
            let (val_loss, val_acc) = self.evaluate(val_x, val_y);
            self.metrics.val_losses.push(val_loss);
            self.metrics.val_accuracies.push(val_acc);

            // Print progress
            println!(
                "Epoch {}/{}: train_loss={:.4}, train_acc={:.4}, val_loss={:.4}, val_acc={:.4}",
                epoch + 1,
                self.config.num_epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc
            );

            // Check for improvement
            if val_loss < self.metrics.best_val_loss {
                self.metrics.best_val_loss = val_loss;
                self.metrics.best_epoch = epoch;
                patience_counter = 0;
            } else {
                patience_counter += 1;

                if patience_counter >= self.config.early_stopping_patience {
                    println!(
                        "\nEarly stopping triggered at epoch {} (patience={})",
                        epoch + 1,
                        self.config.early_stopping_patience
                    );
                    self.metrics.stopped_early = true;
                    break;
                }
            }
        }

        self.metrics.training_time_secs = start_time.elapsed().as_secs_f64();
        self.metrics.early_stop_count = patience_counter;

        println!("\nTraining completed!");
        println!("  Best val_loss: {:.4} at epoch {}", self.metrics.best_val_loss, self.metrics.best_epoch + 1);
        println!("  Training time: {:.2}s", self.metrics.training_time_secs);

        self.metrics.clone()
    }

    /// Get the trained model.
    pub fn get_model(&self) -> &FinancialSentimentModel {
        &self.model
    }

    /// Take ownership of the trained model.
    pub fn into_model(self) -> FinancialSentimentModel {
        self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use ndarray_rand::RandomExt;
    use rand_distr::Normal;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.learning_rate, 1e-4);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.num_epochs, 10);
    }

    #[test]
    fn test_training_config_builder() {
        let config = TrainingConfig::default()
            .with_learning_rate(1e-3)
            .with_batch_size(64)
            .with_epochs(20);

        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.num_epochs, 20);
    }

    #[test]
    fn test_trainer_creation() {
        let model = FinancialSentimentModel::new(768, 12, 8, 16.0, 3);
        let config = TrainingConfig::default();
        let trainer = FineTuningTrainer::new(model, config);

        assert_eq!(trainer.global_step, 0);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let logits = Array1::from_vec(vec![2.0, 1.0, 0.1]);
        let loss = FineTuningTrainer::cross_entropy_loss(&logits, 0);
        assert!(loss > 0.0);
        assert!(loss < 2.0);
    }

    #[test]
    fn test_short_training() {
        let model = FinancialSentimentModel::new(64, 4, 4, 8.0, 3);
        let config = TrainingConfig::default()
            .with_epochs(2)
            .with_batch_size(4);

        let mut trainer = FineTuningTrainer::new(model, config);

        // Create small synthetic dataset
        let normal = Normal::new(0.0, 1.0).unwrap();
        let train_x = Array2::random((20, 64), normal.clone());
        let train_y: Vec<usize> = (0..20).map(|i| i % 3).collect();
        let val_x = Array2::random((10, 64), normal);
        let val_y: Vec<usize> = (0..10).map(|i| i % 3).collect();

        let metrics = trainer.train(&train_x, &train_y, &val_x, &val_y);

        assert_eq!(metrics.train_losses.len(), 2);
        assert_eq!(metrics.val_losses.len(), 2);
    }
}
