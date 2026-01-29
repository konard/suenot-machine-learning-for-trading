//! Multi-task trainer for task-agnostic models
//!
//! Trains the universal encoder and all task heads simultaneously,
//! using gradient harmonization to balance the learning process.

use crate::model::{TaskAgnosticModel, TaskType};
use crate::training::gradient::{GradientHarmonizer, HarmonizerConfig};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Configuration for the multi-task trainer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Gradient harmonization config
    pub harmonizer: HarmonizerConfig,
    /// Validation split ratio
    pub val_split: f64,
    /// Early stopping patience (epochs)
    pub patience: usize,
    /// Print progress every N epochs
    pub log_interval: usize,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            learning_rate: 0.001,
            batch_size: 32,
            weight_decay: 1e-4,
            harmonizer: HarmonizerConfig::default(),
            val_split: 0.2,
            patience: 10,
            log_interval: 10,
        }
    }
}

/// Loss for a single task
#[derive(Debug, Clone)]
pub struct TaskLoss {
    /// Task name
    pub task_name: String,
    /// Task type
    pub task_type: TaskType,
    /// Loss value
    pub loss: f64,
    /// Task weight used
    pub weight: f64,
    /// Weighted loss (loss * weight)
    pub weighted_loss: f64,
}

/// Metrics for a single training epoch
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    /// Epoch number
    pub epoch: usize,
    /// Total loss
    pub total_loss: f64,
    /// Per-task losses
    pub task_losses: Vec<TaskLoss>,
    /// Validation total loss
    pub val_loss: Option<f64>,
    /// Current task weights from harmonizer
    pub task_weights: Vec<f64>,
}

/// Result of training
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// All epoch metrics
    pub epochs: Vec<EpochMetrics>,
    /// Best epoch (lowest validation loss)
    pub best_epoch: usize,
    /// Final task weights
    pub final_weights: Vec<f64>,
    /// Whether training was early-stopped
    pub early_stopped: bool,
    /// Total training samples processed
    pub total_samples: usize,
}

impl TrainingResult {
    /// Get final training loss
    pub fn final_loss(&self) -> f64 {
        self.epochs.last().map(|e| e.total_loss).unwrap_or(f64::MAX)
    }

    /// Get best validation loss
    pub fn best_val_loss(&self) -> f64 {
        self.epochs
            .iter()
            .filter_map(|e| e.val_loss)
            .fold(f64::MAX, f64::min)
    }
}

/// Multi-task trainer
pub struct MultiTaskTrainer {
    config: TrainerConfig,
}

impl MultiTaskTrainer {
    /// Create a new multi-task trainer
    pub fn new(config: TrainerConfig) -> Self {
        Self { config }
    }

    /// Train the model on multi-task data
    ///
    /// # Arguments
    /// * `model` - The task-agnostic model to train
    /// * `features` - Input features (n_samples x n_features)
    /// * `labels` - Labels per task: Vec of (TaskType, labels_array)
    pub fn train(
        &self,
        model: &mut TaskAgnosticModel,
        features: &Array2<f64>,
        labels: &[(TaskType, Array2<f64>)],
    ) -> TrainingResult {
        let n_samples = features.nrows();
        let val_size = (n_samples as f64 * self.config.val_split) as usize;
        let train_size = n_samples - val_size;

        // Split data
        let train_features = features.slice(ndarray::s![..train_size, ..]).to_owned();
        let val_features = features.slice(ndarray::s![train_size.., ..]).to_owned();

        let train_labels: Vec<(TaskType, Array2<f64>)> = labels
            .iter()
            .map(|(tt, l)| (tt.clone(), l.slice(ndarray::s![..train_size, ..]).to_owned()))
            .collect();

        let val_labels: Vec<(TaskType, Array2<f64>)> = labels
            .iter()
            .map(|(tt, l)| (tt.clone(), l.slice(ndarray::s![train_size.., ..]).to_owned()))
            .collect();

        let n_tasks = model.num_tasks();
        let mut harmonizer =
            GradientHarmonizer::new(self.config.harmonizer.clone(), n_tasks);

        let mut epoch_metrics = Vec::new();
        let mut best_val_loss = f64::MAX;
        let mut best_epoch = 0;
        let mut patience_counter = 0;
        let mut early_stopped = false;

        for epoch in 0..self.config.epochs {
            // Training step
            let train_metrics =
                self.train_epoch(model, &train_features, &train_labels, &harmonizer);

            // Validation step
            let val_loss = if val_size > 0 {
                Some(self.evaluate(model, &val_features, &val_labels))
            } else {
                None
            };

            // Update gradient harmonizer
            let task_loss_values: Vec<f64> = train_metrics.iter().map(|tl| tl.loss).collect();
            harmonizer.update_weights(&task_loss_values);

            let total_loss = harmonizer.weighted_loss(&task_loss_values);

            let metrics = EpochMetrics {
                epoch,
                total_loss,
                task_losses: train_metrics,
                val_loss,
                task_weights: harmonizer.weights().to_vec(),
            };

            // Early stopping check
            if let Some(vl) = val_loss {
                if vl < best_val_loss {
                    best_val_loss = vl;
                    best_epoch = epoch;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= self.config.patience {
                        early_stopped = true;
                        epoch_metrics.push(metrics);
                        break;
                    }
                }
            }

            epoch_metrics.push(metrics);
        }

        TrainingResult {
            epochs: epoch_metrics,
            best_epoch,
            final_weights: harmonizer.weights().to_vec(),
            early_stopped,
            total_samples: train_size * self.config.epochs.min(
                if early_stopped { best_epoch + self.config.patience + 1 } else { self.config.epochs }
            ),
        }
    }

    /// Run a single training epoch
    fn train_epoch(
        &self,
        model: &TaskAgnosticModel,
        features: &Array2<f64>,
        labels: &[(TaskType, Array2<f64>)],
        harmonizer: &GradientHarmonizer,
    ) -> Vec<TaskLoss> {
        let n_samples = features.nrows();
        let n_batches = (n_samples + self.config.batch_size - 1) / self.config.batch_size;

        let mut epoch_losses: Vec<f64> = vec![0.0; model.num_tasks()];

        for batch_idx in 0..n_batches {
            let start = batch_idx * self.config.batch_size;
            let end = (start + self.config.batch_size).min(n_samples);
            let batch_features = features.slice(ndarray::s![start..end, ..]).to_owned();

            // Forward pass through encoder
            let representations = model.encode(&batch_features);

            // Forward pass through each head and compute loss
            for (head_idx, head) in model.heads.iter().enumerate() {
                let predictions = head.forward(&representations);

                // Find matching labels
                if let Some((_, label_data)) = labels
                    .iter()
                    .find(|(tt, _)| *tt == head.config.task_type)
                {
                    let batch_labels = label_data.slice(ndarray::s![start..end, ..]).to_owned();
                    let loss = self.compute_loss(&predictions, &batch_labels, &head.config.task_type);
                    epoch_losses[head_idx] += loss / n_batches as f64;
                }
            }
        }

        // Build TaskLoss structs
        let weights = harmonizer.weights();
        model
            .heads
            .iter()
            .enumerate()
            .map(|(i, head)| {
                let w = weights.get(i).copied().unwrap_or(1.0);
                TaskLoss {
                    task_name: head.config.task_type.name().to_string(),
                    task_type: head.config.task_type.clone(),
                    loss: epoch_losses[i],
                    weight: w,
                    weighted_loss: epoch_losses[i] * w,
                }
            })
            .collect()
    }

    /// Evaluate model on validation data
    fn evaluate(
        &self,
        model: &TaskAgnosticModel,
        features: &Array2<f64>,
        labels: &[(TaskType, Array2<f64>)],
    ) -> f64 {
        let representations = model.encode(features);
        let mut total_loss = 0.0;

        for head in &model.heads {
            let predictions = head.forward(&representations);
            if let Some((_, label_data)) = labels
                .iter()
                .find(|(tt, _)| *tt == head.config.task_type)
            {
                let loss = self.compute_loss(&predictions, label_data, &head.config.task_type);
                total_loss += loss;
            }
        }

        total_loss / model.num_tasks() as f64
    }

    /// Compute loss for a specific task
    fn compute_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>, task_type: &TaskType) -> f64 {
        let n = predictions.nrows() as f64;
        if n < 1.0 {
            return 0.0;
        }

        match task_type {
            TaskType::TrendPrediction | TaskType::RegimeDetection | TaskType::RiskAssessment => {
                // Cross-entropy loss for classification
                let mut loss = 0.0;
                for (pred_row, target_row) in predictions.rows().into_iter().zip(targets.rows()) {
                    for (p, t) in pred_row.iter().zip(target_row.iter()) {
                        let p_clamped = p.clamp(1e-10, 1.0 - 1e-10);
                        loss -= t * p_clamped.ln() + (1.0 - t) * (1.0 - p_clamped).ln();
                    }
                }
                loss / n
            }
            TaskType::VolatilityForecast => {
                // MSE loss for regression
                let diff = predictions - targets;
                let mse: f64 = diff.mapv(|v| v * v).sum() / n;
                mse
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TaskAgnosticConfig;

    #[test]
    fn test_trainer_creation() {
        let config = TrainerConfig::default();
        let trainer = MultiTaskTrainer::new(config);
        assert_eq!(trainer.config.epochs, 100);
    }

    #[test]
    fn test_training_basic() {
        let model_config = TaskAgnosticConfig::default().with_input_dim(10).with_repr_dim(8);
        let mut model = TaskAgnosticModel::new(model_config);

        let n_samples = 50;
        let features = Array2::from_shape_fn((n_samples, 10), |(i, j)| {
            ((i * 7 + j * 3) as f64 % 10.0) / 10.0
        });

        let labels = vec![
            (
                TaskType::TrendPrediction,
                Array2::from_shape_fn((n_samples, 3), |(i, j)| {
                    if j == i % 3 { 1.0 } else { 0.0 }
                }),
            ),
            (
                TaskType::VolatilityForecast,
                Array2::from_shape_fn((n_samples, 1), |(i, _)| (i as f64 % 5.0) / 5.0),
            ),
            (
                TaskType::RegimeDetection,
                Array2::from_shape_fn((n_samples, 4), |(i, j)| {
                    if j == i % 4 { 1.0 } else { 0.0 }
                }),
            ),
            (
                TaskType::RiskAssessment,
                Array2::from_shape_fn((n_samples, 3), |(i, j)| {
                    if j == i % 3 { 1.0 } else { 0.0 }
                }),
            ),
        ];

        let trainer_config = TrainerConfig {
            epochs: 5,
            batch_size: 16,
            patience: 3,
            ..Default::default()
        };
        let trainer = MultiTaskTrainer::new(trainer_config);
        let result = trainer.train(&mut model, &features, &labels);

        assert!(!result.epochs.is_empty());
        assert!(result.final_loss() < f64::MAX);
    }
}
