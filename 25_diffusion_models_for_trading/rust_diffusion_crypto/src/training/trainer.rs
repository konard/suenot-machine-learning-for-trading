//! Training loop for DDPM.

use tch::{nn, Tensor, Kind, Device};
use tracing::{info, debug};
use indicatif::{ProgressBar, ProgressStyle};

use crate::model::DDPM;
use crate::data::{TimeSeriesDataset, DataLoader};

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of epochs
    pub epochs: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Gradient clipping value
    pub grad_clip: f64,
    /// Log interval (epochs)
    pub log_interval: usize,
    /// Checkpoint interval (epochs)
    pub checkpoint_interval: usize,
    /// Checkpoint directory
    pub checkpoint_dir: String,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            learning_rate: 1e-4,
            grad_clip: 1.0,
            log_interval: 10,
            checkpoint_interval: 20,
            checkpoint_dir: "checkpoints".to_string(),
        }
    }
}

/// Trainer for DDPM models.
pub struct Trainer {
    model: DDPM,
    config: TrainingConfig,
    optimizer: nn::Optimizer,
    device: Device,
    best_loss: f64,
}

impl Trainer {
    /// Create a new trainer.
    pub fn new(model: DDPM, config: TrainingConfig, device: Device) -> Self {
        let optimizer = nn::Adam::default()
            .build(model.vs(), config.learning_rate)
            .expect("Failed to create optimizer");

        Self {
            model,
            config,
            optimizer,
            device,
            best_loss: f64::INFINITY,
        }
    }

    /// Get reference to the model.
    pub fn model(&self) -> &DDPM {
        &self.model
    }

    /// Get mutable reference to the model.
    pub fn model_mut(&mut self) -> &mut DDPM {
        &mut self.model
    }

    /// Train the model.
    pub fn train(&mut self, train_loader: &mut DataLoader) -> anyhow::Result<Vec<f64>> {
        let mut losses = Vec::new();

        let pb = ProgressBar::new(self.config.epochs as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-"));

        for epoch in 0..self.config.epochs {
            train_loader.reset();

            let mut epoch_losses = Vec::new();

            for (condition, target) in train_loader.by_ref() {
                // Forward pass
                let loss = self.model.compute_loss(&condition, &target);

                // Backward pass
                self.optimizer.zero_grad();
                loss.backward();

                // Gradient clipping
                self.model.vs_mut().variables()
                    .iter()
                    .for_each(|(_, var)| {
                        let _ = var.grad().clamp_(-self.config.grad_clip, self.config.grad_clip);
                    });

                self.optimizer.step();

                let loss_val: f64 = loss.double_value(&[]);
                epoch_losses.push(loss_val);
            }

            let avg_loss = epoch_losses.iter().sum::<f64>() / epoch_losses.len() as f64;
            losses.push(avg_loss);

            // Update best loss
            if avg_loss < self.best_loss {
                self.best_loss = avg_loss;
            }

            // Logging
            if (epoch + 1) % self.config.log_interval == 0 {
                info!(
                    "Epoch {:>4}/{} | Loss: {:.6} | Best: {:.6}",
                    epoch + 1,
                    self.config.epochs,
                    avg_loss,
                    self.best_loss
                );
            }

            // Checkpointing
            if (epoch + 1) % self.config.checkpoint_interval == 0 {
                let checkpoint_path = format!(
                    "{}/ddpm_epoch_{}.pt",
                    self.config.checkpoint_dir,
                    epoch + 1
                );

                if let Err(e) = std::fs::create_dir_all(&self.config.checkpoint_dir) {
                    debug!("Could not create checkpoint dir: {}", e);
                }

                if let Err(e) = self.model.save(&checkpoint_path) {
                    debug!("Could not save checkpoint: {}", e);
                } else {
                    debug!("Saved checkpoint to {}", checkpoint_path);
                }
            }

            pb.set_message(format!("Loss: {:.6}", avg_loss));
            pb.inc(1);
        }

        pb.finish_with_message(format!("Training complete! Best loss: {:.6}", self.best_loss));

        // Save final model
        let final_path = format!("{}/ddpm_final.pt", self.config.checkpoint_dir);
        if let Err(e) = std::fs::create_dir_all(&self.config.checkpoint_dir) {
            debug!("Could not create checkpoint dir: {}", e);
        }
        self.model.save(&final_path)?;
        info!("Model saved to: {}", final_path);

        Ok(losses)
    }

    /// Evaluate the model on a dataset.
    pub fn evaluate(&self, loader: &mut DataLoader) -> f64 {
        loader.reset();

        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for (condition, target) in loader.by_ref() {
            let loss = self.model.compute_loss(&condition, &target);
            total_loss += loss.double_value(&[]);
            num_batches += 1;
        }

        total_loss / num_batches as f64
    }
}

/// Training metrics.
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub learning_rate: f64,
}
