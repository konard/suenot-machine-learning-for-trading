//! Training loop for InceptionTime
//!
//! This module provides the main training loop with early stopping,
//! learning rate scheduling, and checkpointing.

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use tch::{nn, nn::OptimizerConfig, Device, Tensor};
use tracing::{info, warn};

use super::losses::weighted_cross_entropy;
use super::metrics::{accuracy, confusion_matrix, macro_f1, TrainingMetrics};
use crate::data::{DataLoader, TradingDataset};
use crate::model::InceptionTimeNetwork;

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Learning rate scheduler patience
    pub lr_patience: usize,
    /// Learning rate decay factor
    pub lr_decay: f64,
    /// Minimum learning rate
    pub min_lr: f64,
    /// Class weights for loss function
    pub class_weights: Option<Vec<f64>>,
    /// Path to save best model
    pub checkpoint_path: Option<String>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 1500,
            batch_size: 64,
            learning_rate: 0.001,
            weight_decay: 0.0001,
            early_stopping_patience: 100,
            lr_patience: 50,
            lr_decay: 0.5,
            min_lr: 1e-6,
            class_weights: None,
            checkpoint_path: None,
        }
    }
}

/// Trainer for InceptionTime models
pub struct Trainer {
    config: TrainingConfig,
    device: Device,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(config: TrainingConfig) -> Self {
        let device = Device::cuda_if_available();
        info!("Using device: {:?}", device);

        Self { config, device }
    }

    /// Train a single model
    pub fn train(
        &self,
        model: &InceptionTimeNetwork,
        var_store: &mut nn::VarStore,
        train_data: &TradingDataset,
        val_data: &TradingDataset,
    ) -> Result<Vec<TrainingMetrics>> {
        let mut optimizer = nn::Adam::default()
            .weight_decay(self.config.weight_decay)
            .build(var_store, self.config.learning_rate)?;

        let mut metrics_history = Vec::new();
        let mut best_metrics = TrainingMetrics::default();
        let mut patience_counter = 0;
        let mut lr_patience_counter = 0;
        let mut current_lr = self.config.learning_rate;

        // Get class weights
        let class_weights = self.config.class_weights.clone().unwrap_or_else(|| {
            let weights = train_data.class_weights();
            weights.to_vec()
        });

        // Progress bar
        let pb = ProgressBar::new(self.config.epochs as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        for epoch in 0..self.config.epochs {
            let mut epoch_metrics = TrainingMetrics::new(epoch + 1);
            epoch_metrics.learning_rate = current_lr;

            // Training phase
            let train_loss = self.train_epoch(model, &mut optimizer, train_data, &class_weights)?;
            epoch_metrics.train_loss = train_loss;

            // Validation phase
            let (val_loss, val_acc, val_f1) = self.evaluate(model, val_data, &class_weights)?;
            epoch_metrics.val_loss = val_loss;
            epoch_metrics.val_accuracy = val_acc;
            epoch_metrics.val_f1_macro = val_f1;

            // Check for improvement
            if epoch_metrics.improved_over(&best_metrics) {
                best_metrics = epoch_metrics.clone();
                patience_counter = 0;
                lr_patience_counter = 0;

                // Save checkpoint
                if let Some(ref path) = self.config.checkpoint_path {
                    var_store.save(path)?;
                    info!("Saved best model to {}", path);
                }
            } else {
                patience_counter += 1;
                lr_patience_counter += 1;

                // Learning rate decay
                if lr_patience_counter >= self.config.lr_patience && current_lr > self.config.min_lr {
                    current_lr *= self.config.lr_decay;
                    current_lr = current_lr.max(self.config.min_lr);
                    optimizer.set_lr(current_lr);
                    lr_patience_counter = 0;
                    info!("Reduced learning rate to {:.6}", current_lr);
                }
            }

            metrics_history.push(epoch_metrics.clone());
            pb.inc(1);

            // Early stopping
            if patience_counter >= self.config.early_stopping_patience {
                warn!("Early stopping at epoch {}", epoch + 1);
                break;
            }

            // Log progress
            if (epoch + 1) % 50 == 0 {
                info!("{}", epoch_metrics.to_string());
            }
        }

        pb.finish_with_message("Training complete");

        // Load best model if checkpoint exists
        if let Some(ref path) = self.config.checkpoint_path {
            if std::path::Path::new(path).exists() {
                var_store.load(path)?;
                info!("Loaded best model from {}", path);
            }
        }

        info!("Best validation F1: {:.4}", best_metrics.val_f1_macro);

        Ok(metrics_history)
    }

    /// Train for one epoch
    fn train_epoch(
        &self,
        model: &InceptionTimeNetwork,
        optimizer: &mut nn::Optimizer,
        data: &TradingDataset,
        class_weights: &[f64],
    ) -> Result<f64> {
        let mut loader = DataLoader::new(data.len(), self.config.batch_size, true);
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for batch_indices in loader.by_ref() {
            let (features, labels) = data.get_batch(&batch_indices);

            // Convert to tensors
            let x = Tensor::try_from(features)?.to_device(self.device);
            let y = Tensor::from_slice(&labels).to_device(self.device);

            // Forward pass
            let logits = model.forward(&x, true);

            // Compute loss
            let loss = weighted_cross_entropy(&logits, &y, class_weights);

            // Backward pass
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += f64::try_from(&loss)?;
            num_batches += 1;
        }

        Ok(total_loss / num_batches as f64)
    }

    /// Evaluate on validation set
    fn evaluate(
        &self,
        model: &InceptionTimeNetwork,
        data: &TradingDataset,
        class_weights: &[f64],
    ) -> Result<(f64, f64, f64)> {
        let mut loader = DataLoader::new(data.len(), self.config.batch_size, false);
        let mut total_loss = 0.0;
        let mut all_preds = Vec::new();
        let mut all_labels = Vec::new();

        tch::no_grad(|| {
            for batch_indices in loader.by_ref() {
                let (features, labels) = data.get_batch(&batch_indices);

                let x = Tensor::try_from(features).unwrap().to_device(self.device);
                let y = Tensor::from_slice(&labels).to_device(self.device);

                let logits = model.forward(&x, false);
                let loss = weighted_cross_entropy(&logits, &y, class_weights);

                total_loss += f64::try_from(&loss).unwrap_or(0.0);

                let preds = logits.argmax(-1, false);
                all_preds.extend(Vec::<i64>::try_from(&preds).unwrap_or_default());
                all_labels.extend(labels);
            }
        });

        let num_batches = (data.len() + self.config.batch_size - 1) / self.config.batch_size;
        let avg_loss = total_loss / num_batches as f64;

        // Calculate metrics
        let preds_tensor = Tensor::from_slice(&all_preds);
        let labels_tensor = Tensor::from_slice(&all_labels);

        let acc = accuracy(
            &preds_tensor.unsqueeze(-1).onehot(3),
            &labels_tensor,
        );

        let confusion = confusion_matrix(
            &preds_tensor.unsqueeze(-1).onehot(3),
            &labels_tensor,
            3,
        );
        let f1 = macro_f1(&confusion);

        Ok((avg_loss, acc, f1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.epochs, 1500);
        assert_eq!(config.batch_size, 64);
    }
}
