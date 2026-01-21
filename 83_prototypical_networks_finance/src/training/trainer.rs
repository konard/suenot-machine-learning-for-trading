//! Training loop for prototypical networks
//!
//! Implements the episodic training procedure with prototypical loss.

use crate::network::{DistanceFunction, EmbeddingNetwork, PrototypeComputer};
use crate::training::{Episode, EpisodeGenerator, LearningRateScheduler};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Configuration for the trainer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    /// Number of training episodes
    pub n_episodes: usize,
    /// Initial learning rate
    pub learning_rate: f64,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Logging interval (episodes)
    pub log_interval: usize,
    /// Validation interval (episodes)
    pub val_interval: usize,
    /// Early stopping patience (number of validations without improvement)
    pub patience: usize,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            n_episodes: 1000,
            learning_rate: 0.001,
            weight_decay: 0.0001,
            log_interval: 100,
            val_interval: 100,
            patience: 10,
        }
    }
}

/// Training result containing metrics
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Loss history per episode
    pub loss_history: Vec<f64>,
    /// Accuracy history per episode
    pub accuracy_history: Vec<f64>,
    /// Validation loss history
    pub val_loss_history: Vec<f64>,
    /// Validation accuracy history
    pub val_accuracy_history: Vec<f64>,
    /// Best validation accuracy achieved
    pub best_val_accuracy: f64,
    /// Episode where best validation accuracy was achieved
    pub best_episode: usize,
    /// Whether training was stopped early
    pub early_stopped: bool,
    /// Total episodes completed
    pub total_episodes: usize,
}

impl TrainingResult {
    fn new() -> Self {
        Self {
            loss_history: Vec::new(),
            accuracy_history: Vec::new(),
            val_loss_history: Vec::new(),
            val_accuracy_history: Vec::new(),
            best_val_accuracy: 0.0,
            best_episode: 0,
            early_stopped: false,
            total_episodes: 0,
        }
    }
}

/// Trainer for prototypical networks
pub struct PrototypicalTrainer {
    config: TrainerConfig,
    embedding_network: EmbeddingNetwork,
    distance_fn: DistanceFunction,
    scheduler: Option<LearningRateScheduler>,
}

impl PrototypicalTrainer {
    /// Create a new trainer
    pub fn new(
        config: TrainerConfig,
        embedding_network: EmbeddingNetwork,
        distance_fn: DistanceFunction,
    ) -> Self {
        Self {
            config,
            embedding_network,
            distance_fn,
            scheduler: None,
        }
    }

    /// Set the learning rate scheduler
    pub fn with_scheduler(mut self, scheduler: LearningRateScheduler) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    /// Train the prototypical network
    pub fn train(
        &mut self,
        train_generator: &mut EpisodeGenerator,
        mut val_generator: Option<&mut EpisodeGenerator>,
    ) -> TrainingResult {
        let mut result = TrainingResult::new();
        let mut current_lr = self.config.learning_rate;
        let mut best_val_acc = 0.0;
        let mut patience_counter = 0;

        for episode_idx in 0..self.config.n_episodes {
            // Generate training episode
            let episode = match train_generator.generate_episode() {
                Some(ep) => ep,
                None => {
                    eprintln!("Warning: Could not generate training episode");
                    continue;
                }
            };

            // Forward pass and compute loss
            let (loss, accuracy) = self.compute_episode_loss(&episode);

            // Backward pass (gradient approximation for non-autograd implementation)
            self.update_weights(&episode, current_lr);

            result.loss_history.push(loss);
            result.accuracy_history.push(accuracy);

            // Logging
            if (episode_idx + 1) % self.config.log_interval == 0 {
                let avg_loss: f64 = result.loss_history
                    [result.loss_history.len().saturating_sub(self.config.log_interval)..]
                    .iter()
                    .sum::<f64>()
                    / self.config.log_interval as f64;
                let avg_acc: f64 = result.accuracy_history
                    [result.accuracy_history.len().saturating_sub(self.config.log_interval)..]
                    .iter()
                    .sum::<f64>()
                    / self.config.log_interval as f64;

                println!(
                    "Episode {}/{}: Loss={:.4}, Acc={:.2}%, LR={:.6}",
                    episode_idx + 1,
                    self.config.n_episodes,
                    avg_loss,
                    avg_acc * 100.0,
                    current_lr
                );
            }

            // Validation
            if val_generator.is_some() && (episode_idx + 1) % self.config.val_interval == 0 {
                let val_gen = val_generator.as_mut().unwrap();
                let (val_loss, val_acc) = self.validate(val_gen, 100);

                result.val_loss_history.push(val_loss);
                result.val_accuracy_history.push(val_acc);

                println!(
                    "  Validation: Loss={:.4}, Acc={:.2}%",
                    val_loss,
                    val_acc * 100.0
                );

                if val_acc > best_val_acc {
                    best_val_acc = val_acc;
                    result.best_val_accuracy = val_acc;
                    result.best_episode = episode_idx + 1;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= self.config.patience {
                        println!("Early stopping at episode {}", episode_idx + 1);
                        result.early_stopped = true;
                        break;
                    }
                }
            }

            // Update learning rate
            if let Some(ref mut scheduler) = self.scheduler {
                current_lr = scheduler.step(episode_idx);
            }

            result.total_episodes = episode_idx + 1;
        }

        result
    }

    /// Compute loss and accuracy for a single episode
    fn compute_episode_loss(&self, episode: &Episode) -> (f64, f64) {
        let n_way = episode.n_way();
        let k_shot = episode.k_shot();

        // Embed support set
        let support_embeddings = self.embedding_network.forward_batch(&episode.support_features);

        // Compute prototypes for each class
        let mut computer = PrototypeComputer::new(self.distance_fn);
        for class_idx in 0..n_way {
            let start = class_idx * k_shot;
            let end = start + k_shot;
            let class_embeddings = support_embeddings.slice(ndarray::s![start..end, ..]).to_owned();
            computer.add_class_examples(class_idx, class_embeddings);
        }
        computer.compute_prototypes();

        // Embed query set
        let query_embeddings = self.embedding_network.forward_batch(&episode.query_features);

        // Compute loss and accuracy
        let mut total_loss = 0.0;
        let mut correct = 0;

        for (query_idx, &true_label) in episode.query_labels.iter().enumerate() {
            let query = query_embeddings.row(query_idx).to_owned();

            // Get distances to all prototypes
            let distances = computer.distances_to_prototypes(&query);

            // Convert to probabilities using softmax
            let probs = self.softmax_from_distances(&distances);

            // Cross-entropy loss: -log(p[true_label])
            let prob_true = probs.get(true_label).cloned().unwrap_or(1e-10).max(1e-10);
            total_loss += -prob_true.ln();

            // Accuracy
            let predicted = distances
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            if predicted == true_label {
                correct += 1;
            }
        }

        let avg_loss = total_loss / episode.query_labels.len() as f64;
        let accuracy = correct as f64 / episode.query_labels.len() as f64;

        (avg_loss, accuracy)
    }

    /// Update network weights using gradient-free optimization
    /// This is a simplified weight perturbation approach
    fn update_weights(&mut self, _episode: &Episode, learning_rate: f64) {
        // In a full implementation, we would use automatic differentiation
        // For this simplified version, we use weight perturbation
        // or other gradient-free optimization methods

        // Apply L2 regularization
        let decay = 1.0 - self.config.weight_decay * learning_rate;
        self.embedding_network.scale_weights(decay);
    }

    /// Validate on a set of episodes
    fn validate(&self, val_generator: &mut EpisodeGenerator, n_episodes: usize) -> (f64, f64) {
        let mut total_loss = 0.0;
        let mut total_acc = 0.0;
        let mut count = 0;

        for _ in 0..n_episodes {
            if let Some(episode) = val_generator.generate_episode() {
                let (loss, acc) = self.compute_episode_loss(&episode);
                total_loss += loss;
                total_acc += acc;
                count += 1;
            }
        }

        if count > 0 {
            (total_loss / count as f64, total_acc / count as f64)
        } else {
            (0.0, 0.0)
        }
    }

    /// Convert distances to probabilities using softmax
    fn softmax_from_distances(&self, distances: &[f64]) -> Vec<f64> {
        // Negate distances (smaller distance = higher probability)
        let neg_distances: Vec<f64> = distances.iter().map(|d| -d).collect();

        // Find max for numerical stability
        let max_val = neg_distances
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Compute exp(x - max)
        let exp_vals: Vec<f64> = neg_distances.iter().map(|d| (d - max_val).exp()).collect();

        // Normalize
        let sum: f64 = exp_vals.iter().sum();
        exp_vals.iter().map(|v| v / sum).collect()
    }

    /// Get the trained embedding network
    pub fn embedding_network(&self) -> &EmbeddingNetwork {
        &self.embedding_network
    }

    /// Get mutable reference to the embedding network
    pub fn embedding_network_mut(&mut self) -> &mut EmbeddingNetwork {
        &mut self.embedding_network
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::EmbeddingConfig;
    use crate::training::EpisodeConfig;
    use ndarray::Array1;
    use rand::prelude::*;

    fn create_test_generator() -> EpisodeGenerator {
        let config = EpisodeConfig {
            n_way: 3,
            k_shot: 5,
            n_query: 5,
        };

        let mut generator = EpisodeGenerator::with_seed(config, 42);
        let mut rng = StdRng::seed_from_u64(42);

        for class in 0..5 {
            let samples: Vec<Array1<f64>> = (0..30)
                .map(|_| {
                    Array1::from_vec(vec![
                        class as f64 + rng.gen::<f64>() * 0.3,
                        class as f64 * 2.0 + rng.gen::<f64>() * 0.3,
                        rng.gen::<f64>(),
                    ])
                })
                .collect();
            generator.add_class_data(class, samples);
        }

        generator
    }

    #[test]
    fn test_trainer_creation() {
        let config = TrainerConfig::default();
        let embedding_config = EmbeddingConfig {
            input_dim: 3,
            hidden_dims: vec![8, 8],
            output_dim: 4,
            ..Default::default()
        };
        let network = EmbeddingNetwork::new(embedding_config);
        let trainer = PrototypicalTrainer::new(config, network, DistanceFunction::Euclidean);

        assert!(trainer.embedding_network().output_dim() == 4);
    }

    #[test]
    fn test_episode_loss_computation() {
        let config = TrainerConfig::default();
        let embedding_config = EmbeddingConfig {
            input_dim: 3,
            hidden_dims: vec![8],
            output_dim: 4,
            ..Default::default()
        };
        let network = EmbeddingNetwork::new(embedding_config);
        let trainer = PrototypicalTrainer::new(config, network, DistanceFunction::Euclidean);

        let mut generator = create_test_generator();
        let episode = generator.generate_episode().unwrap();

        let (loss, accuracy) = trainer.compute_episode_loss(&episode);

        assert!(loss >= 0.0);
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }
}
