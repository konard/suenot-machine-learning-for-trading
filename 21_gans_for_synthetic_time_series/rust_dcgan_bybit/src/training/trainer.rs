//! Training loop implementation for DCGAN
//!
//! Provides the main training loop with proper alternating updates
//! for generator and discriminator.

use indicatif::{ProgressBar, ProgressStyle};
use tch::{nn, nn::ModuleT, Device, Tensor};
use tracing::{info, warn};

use crate::data::DataLoader;
use crate::model::DCGAN;
use super::losses::{discriminator_loss, generator_loss};
use super::metrics::TrainingMetrics;

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Learning rate for generator
    pub gen_lr: f64,
    /// Learning rate for discriminator
    pub disc_lr: f64,
    /// Number of discriminator updates per generator update
    pub disc_steps: usize,
    /// Save checkpoint every N epochs
    pub checkpoint_every: usize,
    /// Directory to save checkpoints
    pub checkpoint_dir: String,
    /// Whether to use label smoothing
    pub label_smoothing: bool,
    /// Smooth label for real samples (e.g., 0.9)
    pub smooth_real: f64,
    /// Smooth label for fake samples (e.g., 0.1)
    pub smooth_fake: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            gen_lr: 2e-4,
            disc_lr: 2e-4,
            disc_steps: 1,
            checkpoint_every: 10,
            checkpoint_dir: "checkpoints".to_string(),
            label_smoothing: false,
            smooth_real: 0.9,
            smooth_fake: 0.1,
        }
    }
}

/// DCGAN Trainer
pub struct Trainer {
    config: TrainingConfig,
    device: Device,
    metrics: TrainingMetrics,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(config: TrainingConfig, device: Device) -> Self {
        Self {
            config,
            device,
            metrics: TrainingMetrics::new(),
        }
    }

    /// Train the DCGAN model
    ///
    /// # Arguments
    ///
    /// * `model` - DCGAN model to train
    /// * `data_loader` - DataLoader providing training batches
    ///
    /// # Returns
    ///
    /// Training metrics
    pub fn train(&mut self, model: &mut DCGAN, data_loader: &mut DataLoader) -> &TrainingMetrics {
        let mut gen_opt = model.gen_optimizer(self.config.gen_lr);
        let mut disc_opt = model.disc_optimizer(self.config.disc_lr);

        let latent_dim = model.latent_dim();
        let num_batches = data_loader.num_batches();

        info!(
            "Starting training for {} epochs, {} batches per epoch",
            self.config.epochs, num_batches
        );

        // Create checkpoint directory
        std::fs::create_dir_all(&self.config.checkpoint_dir).ok();

        for epoch in 0..self.config.epochs {
            let mut epoch_gen_loss = 0.0;
            let mut epoch_disc_loss = 0.0;
            let mut epoch_real_acc = 0.0;
            let mut epoch_fake_acc = 0.0;
            let mut batch_count = 0;

            // Progress bar for epoch
            let pb = ProgressBar::new(num_batches as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                    .unwrap()
                    .progress_chars("##-"),
            );

            // Iterate over batches
            for real_batch in data_loader.iter() {
                let batch_size = real_batch.size()[0];

                // Convert to tensor on device
                let real_data = Tensor::try_from(real_batch)
                    .unwrap()
                    .to_device(self.device);

                // ========== Train Discriminator ==========
                for _ in 0..self.config.disc_steps {
                    // Generate fake data
                    let noise = Tensor::randn(
                        [batch_size, latent_dim],
                        (tch::Kind::Float, self.device),
                    );
                    let fake_data = model.generator.forward_t(&noise, true);

                    // Discriminator predictions
                    let real_output = model.discriminator.forward_t(&real_data, true);
                    let fake_output = model.discriminator.forward_t(&fake_data.detach(), true);

                    // Calculate discriminator loss
                    let d_loss = if self.config.label_smoothing {
                        super::losses::discriminator_loss_smoothed(
                            &real_output,
                            &fake_output,
                            self.config.smooth_real,
                            self.config.smooth_fake,
                        )
                    } else {
                        discriminator_loss(&real_output, &fake_output)
                    };

                    // Update discriminator
                    disc_opt.zero_grad();
                    d_loss.backward();
                    disc_opt.step();

                    epoch_disc_loss += d_loss.double_value(&[]);

                    // Calculate accuracies
                    let real_acc = real_output.sigmoid().ge(0.5).to_kind(tch::Kind::Float).mean(tch::Kind::Float);
                    let fake_acc = fake_output.sigmoid().lt(0.5).to_kind(tch::Kind::Float).mean(tch::Kind::Float);
                    epoch_real_acc += real_acc.double_value(&[]);
                    epoch_fake_acc += fake_acc.double_value(&[]);
                }

                // ========== Train Generator ==========
                let noise = Tensor::randn(
                    [batch_size, latent_dim],
                    (tch::Kind::Float, self.device),
                );
                let fake_data = model.generator.forward_t(&noise, true);
                let fake_output = model.discriminator.forward_t(&fake_data, true);

                let g_loss = generator_loss(&fake_output);

                gen_opt.zero_grad();
                g_loss.backward();
                gen_opt.step();

                epoch_gen_loss += g_loss.double_value(&[]);
                batch_count += 1;

                // Update progress bar
                pb.set_message(format!(
                    "G: {:.4}, D: {:.4}",
                    g_loss.double_value(&[]),
                    epoch_disc_loss / (batch_count * self.config.disc_steps) as f64
                ));
                pb.inc(1);
            }

            pb.finish_with_message("done");

            // Calculate epoch averages
            let total_disc_updates = (batch_count * self.config.disc_steps) as f64;
            let avg_gen_loss = epoch_gen_loss / batch_count as f64;
            let avg_disc_loss = epoch_disc_loss / total_disc_updates;
            let avg_real_acc = epoch_real_acc / total_disc_updates;
            let avg_fake_acc = epoch_fake_acc / total_disc_updates;

            // Record metrics
            self.metrics.record_epoch(avg_gen_loss, avg_disc_loss, avg_real_acc, avg_fake_acc);

            info!(
                "Epoch {}/{}: G_loss={:.4}, D_loss={:.4}, Real_acc={:.2}%, Fake_acc={:.2}%",
                epoch + 1,
                self.config.epochs,
                avg_gen_loss,
                avg_disc_loss,
                avg_real_acc * 100.0,
                avg_fake_acc * 100.0
            );

            // Check for mode collapse
            if self.metrics.check_mode_collapse(10) {
                warn!("Possible mode collapse detected! Consider adjusting learning rates.");
            }

            // Save checkpoint
            if (epoch + 1) % self.config.checkpoint_every == 0 {
                let gen_path = format!("{}/generator_epoch_{}.pt", self.config.checkpoint_dir, epoch + 1);
                let disc_path = format!("{}/discriminator_epoch_{}.pt", self.config.checkpoint_dir, epoch + 1);

                if let Err(e) = model.save(&gen_path, &disc_path) {
                    warn!("Failed to save checkpoint: {}", e);
                } else {
                    info!("Saved checkpoint at epoch {}", epoch + 1);
                }
            }

            data_loader.reset();
        }

        // Save final model
        let gen_path = format!("{}/generator_final.pt", self.config.checkpoint_dir);
        let disc_path = format!("{}/discriminator_final.pt", self.config.checkpoint_dir);
        if let Err(e) = model.save(&gen_path, &disc_path) {
            warn!("Failed to save final model: {}", e);
        }

        // Save metrics
        let metrics_path = format!("{}/training_metrics.csv", self.config.checkpoint_dir);
        if let Err(e) = self.metrics.save_csv(&metrics_path) {
            warn!("Failed to save metrics: {}", e);
        }

        &self.metrics
    }

    /// Get training metrics
    pub fn metrics(&self) -> &TrainingMetrics {
        &self.metrics
    }

    /// Get configuration
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }
}

/// Single training step (for more fine-grained control)
pub fn train_step(
    model: &mut DCGAN,
    real_data: &Tensor,
    gen_opt: &mut nn::Optimizer,
    disc_opt: &mut nn::Optimizer,
) -> (f64, f64) {
    let batch_size = real_data.size()[0];
    let latent_dim = model.latent_dim();
    let device = model.device;

    // Train discriminator
    let noise = Tensor::randn([batch_size, latent_dim], (tch::Kind::Float, device));
    let fake_data = model.generator.forward_t(&noise, true);

    let real_output = model.discriminator.forward_t(real_data, true);
    let fake_output = model.discriminator.forward_t(&fake_data.detach(), true);

    let d_loss = discriminator_loss(&real_output, &fake_output);

    disc_opt.zero_grad();
    d_loss.backward();
    disc_opt.step();

    // Train generator
    let noise = Tensor::randn([batch_size, latent_dim], (tch::Kind::Float, device));
    let fake_data = model.generator.forward_t(&noise, true);
    let fake_output = model.discriminator.forward_t(&fake_data, true);

    let g_loss = generator_loss(&fake_output);

    gen_opt.zero_grad();
    g_loss.backward();
    gen_opt.step();

    (g_loss.double_value(&[]), d_loss.double_value(&[]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.epochs, 100);
        assert_eq!(config.disc_steps, 1);
    }
}
