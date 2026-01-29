//! Transfer learning trainer for pre-training and fine-tuning.

use ndarray::Array2;
use crate::network::{TransferNetwork, AdaptationMethod, FineTuneStrategy};

/// Configuration for pre-training phase.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate for pre-training
    pub learning_rate: f64,
    /// Number of pre-training epochs
    pub epochs: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Early stopping patience (epochs without improvement)
    pub patience: usize,
    /// Minimum improvement for early stopping
    pub min_delta: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            epochs: 100,
            batch_size: 32,
            patience: 10,
            min_delta: 1e-4,
        }
    }
}

/// Configuration for domain adaptation phase.
#[derive(Debug, Clone)]
pub struct AdaptationConfig {
    /// Learning rate for fine-tuning
    pub learning_rate: f64,
    /// Number of adaptation epochs
    pub epochs: usize,
    /// Domain adaptation method
    pub method: AdaptationMethod,
    /// Fine-tuning strategy
    pub strategy: FineTuneStrategy,
    /// Weight of adaptation loss
    pub adaptation_weight: f64,
    /// Maximum MMD threshold to stop adaptation
    pub mmd_threshold: f64,
}

impl Default for AdaptationConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.0001,
            epochs: 50,
            method: AdaptationMethod::MMD,
            strategy: FineTuneStrategy::PartialFineTune,
            adaptation_weight: 1.0,
            mmd_threshold: 0.1,
        }
    }
}

/// Result of a training/adaptation phase.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Loss history per epoch
    pub loss_history: Vec<f64>,
    /// Accuracy history per epoch (if labels provided)
    pub accuracy_history: Vec<f64>,
    /// Final loss value
    pub final_loss: f64,
    /// Number of epochs actually trained
    pub epochs_trained: usize,
    /// Whether early stopping was triggered
    pub early_stopped: bool,
}

/// Transfer learning trainer managing the full training pipeline.
#[derive(Debug)]
pub struct TransferTrainer {
    /// The transfer learning network
    network: TransferNetwork,
    /// Pre-training configuration
    pretrain_config: TrainingConfig,
    /// Adaptation configuration
    adapt_config: AdaptationConfig,
}

impl TransferTrainer {
    /// Create a new trainer with a network.
    pub fn new(network: TransferNetwork) -> Self {
        Self {
            network,
            pretrain_config: TrainingConfig::default(),
            adapt_config: AdaptationConfig::default(),
        }
    }

    /// Set the pre-training configuration.
    pub fn with_pretrain_config(mut self, config: TrainingConfig) -> Self {
        self.pretrain_config = config;
        self
    }

    /// Set the adaptation configuration.
    pub fn with_adapt_config(mut self, config: AdaptationConfig) -> Self {
        self.adapt_config = config;
        self
    }

    /// Pre-train the network on source domain data.
    ///
    /// # Arguments
    /// * `source_features` - Source domain feature matrix
    /// * `source_labels` - Source domain labels
    pub fn pretrain(
        &mut self,
        source_features: &Array2<f64>,
        source_labels: &[usize],
    ) -> TrainingResult {
        let config = self.pretrain_config.clone();
        let mut loss_history = Vec::new();
        let mut accuracy_history = Vec::new();
        let mut best_loss = f64::MAX;
        let mut patience_counter = 0;

        for epoch in 0..config.epochs {
            // Train for one epoch
            self.network.pretrain(
                source_features,
                source_labels,
                config.learning_rate,
                1,
            );

            // Compute loss and accuracy
            let predictions = self.network.predict(source_features);
            let correct = predictions
                .iter()
                .zip(source_labels.iter())
                .filter(|(&pred, &label)| pred == label)
                .count();
            let accuracy = correct as f64 / source_labels.len().max(1) as f64;

            // Approximate loss from accuracy
            let loss = 1.0 - accuracy;
            loss_history.push(loss);
            accuracy_history.push(accuracy);

            // Early stopping check
            if loss < best_loss - config.min_delta {
                best_loss = loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
            }

            if patience_counter >= config.patience {
                tracing::info!("Early stopping at epoch {}", epoch + 1);
                return TrainingResult {
                    loss_history,
                    accuracy_history,
                    final_loss: loss,
                    epochs_trained: epoch + 1,
                    early_stopped: true,
                };
            }
        }

        let final_loss = loss_history.last().copied().unwrap_or(0.0);

        TrainingResult {
            loss_history,
            accuracy_history,
            final_loss,
            epochs_trained: config.epochs,
            early_stopped: false,
        }
    }

    /// Adapt the network to target domain using domain adaptation.
    ///
    /// # Arguments
    /// * `source_features` - Source domain features (for computing adaptation loss)
    /// * `target_features` - Target domain features
    /// * `target_labels` - Optional target domain labels (for supervised adaptation)
    pub fn adapt(
        &mut self,
        source_features: &Array2<f64>,
        target_features: &Array2<f64>,
        target_labels: Option<&[usize]>,
    ) -> TrainingResult {
        let config = self.adapt_config.clone();

        // Apply fine-tuning strategy
        self.network = self.network.clone()
            .with_fine_tune_strategy(config.strategy)
            .with_adaptation_method(config.method);
        self.network.apply_fine_tune_strategy();

        // Fit domain adapter
        let source_feat = self.network.feature_extractor().forward(source_features);
        let target_feat = self.network.feature_extractor().forward(target_features);
        self.network.domain_adapter_mut().fit(&source_feat, &target_feat);

        let mut loss_history = Vec::new();
        let mut accuracy_history = Vec::new();
        let mut best_loss = f64::MAX;
        let mut patience_counter = 0;

        for epoch in 0..config.epochs {
            // If target labels available, do supervised fine-tuning
            if let Some(labels) = target_labels {
                self.network.pretrain(
                    target_features,
                    labels,
                    config.learning_rate,
                    1,
                );

                let predictions = self.network.predict(target_features);
                let correct = predictions
                    .iter()
                    .zip(labels.iter())
                    .filter(|(&pred, &label)| pred == label)
                    .count();
                let accuracy = correct as f64 / labels.len().max(1) as f64;
                accuracy_history.push(accuracy);
            }

            // Compute adaptation loss
            let src_feat = self.network.feature_extractor().forward(source_features);
            let tgt_feat = self.network.feature_extractor().forward(target_features);
            let adapt_loss = self.network.domain_adapter().compute_loss(&src_feat, &tgt_feat);
            loss_history.push(adapt_loss);

            // Check MMD convergence
            if adapt_loss < config.mmd_threshold {
                tracing::info!("Adaptation converged at epoch {} (loss: {:.6})", epoch + 1, adapt_loss);
                return TrainingResult {
                    loss_history,
                    accuracy_history,
                    final_loss: adapt_loss,
                    epochs_trained: epoch + 1,
                    early_stopped: true,
                };
            }

            // Early stopping
            if adapt_loss < best_loss - self.pretrain_config.min_delta {
                best_loss = adapt_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
            }

            if patience_counter >= self.pretrain_config.patience {
                return TrainingResult {
                    loss_history,
                    accuracy_history,
                    final_loss: adapt_loss,
                    epochs_trained: epoch + 1,
                    early_stopped: true,
                };
            }
        }

        let final_loss = loss_history.last().copied().unwrap_or(0.0);

        TrainingResult {
            loss_history,
            accuracy_history,
            final_loss,
            epochs_trained: config.epochs,
            early_stopped: false,
        }
    }

    /// Run the full transfer learning pipeline.
    ///
    /// 1. Pre-train on source domain
    /// 2. Adapt to target domain
    /// 3. Return the trained network
    pub fn run_pipeline(
        &mut self,
        source_features: &Array2<f64>,
        source_labels: &[usize],
        target_features: &Array2<f64>,
        target_labels: Option<&[usize]>,
    ) -> (TrainingResult, TrainingResult) {
        let pretrain_result = self.pretrain(source_features, source_labels);
        let adapt_result = self.adapt(source_features, target_features, target_labels);
        (pretrain_result, adapt_result)
    }

    /// Get a reference to the trained network.
    pub fn network(&self) -> &TransferNetwork {
        &self.network
    }

    /// Take ownership of the trained network.
    pub fn into_network(self) -> TransferNetwork {
        self.network
    }

    /// Compute domain similarity between source and target.
    pub fn compute_domain_similarity(
        &self,
        source_features: &Array2<f64>,
        target_features: &Array2<f64>,
    ) -> f64 {
        let src_feat = self.network.feature_extractor().forward(source_features);
        let tgt_feat = self.network.feature_extractor().forward(target_features);
        self.network.domain_adapter().domain_similarity(&src_feat, &tgt_feat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use ndarray_rand::RandomExt;
    use rand_distr::Normal;

    fn create_test_data() -> (Array2<f64>, Vec<usize>, Array2<f64>, Vec<usize>) {
        let source = Array2::random((100, 20), Normal::new(0.0, 1.0).unwrap());
        let source_labels: Vec<usize> = (0..100).map(|i| i % 3).collect();
        let target = Array2::random((50, 20), Normal::new(1.0, 1.5).unwrap());
        let target_labels: Vec<usize> = (0..50).map(|i| i % 3).collect();
        (source, source_labels, target, target_labels)
    }

    #[test]
    fn test_trainer_creation() {
        let network = TransferNetwork::new(20, 128, 64, true);
        let trainer = TransferTrainer::new(network);
        assert_eq!(trainer.network().num_classes(), 3);
    }

    #[test]
    fn test_pretrain() {
        let (source, source_labels, _, _) = create_test_data();
        let network = TransferNetwork::new(20, 64, 32, false);
        let mut trainer = TransferTrainer::new(network)
            .with_pretrain_config(TrainingConfig {
                epochs: 5,
                learning_rate: 0.01,
                ..Default::default()
            });

        let result = trainer.pretrain(&source, &source_labels);
        assert!(!result.loss_history.is_empty());
        assert_eq!(result.accuracy_history.len(), result.loss_history.len());
    }

    #[test]
    fn test_adapt() {
        let (source, source_labels, target, target_labels) = create_test_data();
        let network = TransferNetwork::new(20, 64, 32, true);
        let mut trainer = TransferTrainer::new(network)
            .with_pretrain_config(TrainingConfig {
                epochs: 3,
                ..Default::default()
            })
            .with_adapt_config(AdaptationConfig {
                epochs: 3,
                ..Default::default()
            });

        trainer.pretrain(&source, &source_labels);
        let result = trainer.adapt(&source, &target, Some(&target_labels));
        assert!(!result.loss_history.is_empty());
    }

    #[test]
    fn test_full_pipeline() {
        let (source, source_labels, target, target_labels) = create_test_data();
        let network = TransferNetwork::new(20, 32, 16, true);
        let mut trainer = TransferTrainer::new(network)
            .with_pretrain_config(TrainingConfig {
                epochs: 3,
                learning_rate: 0.01,
                ..Default::default()
            })
            .with_adapt_config(AdaptationConfig {
                epochs: 3,
                ..Default::default()
            });

        let (pretrain_result, adapt_result) = trainer.run_pipeline(
            &source,
            &source_labels,
            &target,
            Some(&target_labels),
        );

        assert!(!pretrain_result.loss_history.is_empty());
        assert!(!adapt_result.loss_history.is_empty());
    }

    #[test]
    fn test_domain_similarity() {
        let (source, _, target, _) = create_test_data();
        let network = TransferNetwork::new(20, 32, 16, true);
        let trainer = TransferTrainer::new(network);
        let similarity = trainer.compute_domain_similarity(&source, &target);
        assert!(similarity >= 0.0 && similarity <= 1.0);
    }

    #[test]
    fn test_config_defaults() {
        let config = TrainingConfig::default();
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 32);

        let adapt = AdaptationConfig::default();
        assert_eq!(adapt.method, AdaptationMethod::MMD);
        assert_eq!(adapt.strategy, FineTuneStrategy::PartialFineTune);
    }
}
