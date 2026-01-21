//! Episodic training for Matching Networks
//!
//! Implements the episodic training paradigm where each training step
//! simulates a few-shot learning scenario.

use crate::network::MatchingNetwork;
use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use std::collections::HashMap;

/// Configuration for episodic training
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of classes per episode (N-way)
    pub n_way: usize,
    /// Number of support examples per class (K-shot)
    pub k_shot: usize,
    /// Number of query examples per class
    pub n_query: usize,
    /// Number of training episodes
    pub num_episodes: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Validation frequency (episodes)
    pub val_frequency: usize,
    /// Early stopping patience
    pub patience: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            n_way: 5,
            k_shot: 5,
            n_query: 5,
            num_episodes: 1000,
            learning_rate: 0.001,
            val_frequency: 100,
            patience: 10,
        }
    }
}

/// A single training episode
#[derive(Debug, Clone)]
pub struct Episode {
    /// Support set features [n_way * k_shot, feature_dim]
    pub support_features: Array2<f64>,
    /// Support set labels [n_way * k_shot]
    pub support_labels: Array1<usize>,
    /// Query set features [n_way * n_query, feature_dim]
    pub query_features: Array2<f64>,
    /// Query set labels [n_way * n_query]
    pub query_labels: Array1<usize>,
    /// Classes included in this episode
    pub classes: Vec<usize>,
}

/// Training metrics
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// Training accuracy per episode
    pub train_accuracy: Vec<f64>,
    /// Validation accuracy (at val_frequency intervals)
    pub val_accuracy: Vec<f64>,
    /// Training loss per episode
    pub train_loss: Vec<f64>,
    /// Best validation accuracy achieved
    pub best_val_accuracy: f64,
    /// Episode at which best accuracy was achieved
    pub best_episode: usize,
}

/// Episodic trainer for Matching Networks
pub struct EpisodicTrainer {
    /// Training configuration
    config: TrainingConfig,
    /// Training data grouped by class
    train_data: HashMap<usize, Vec<Array1<f64>>>,
    /// Validation data grouped by class
    val_data: HashMap<usize, Vec<Array1<f64>>>,
    /// Random number generator
    rng: rand::rngs::ThreadRng,
}

impl EpisodicTrainer {
    /// Create a new episodic trainer
    ///
    /// # Arguments
    /// * `config` - Training configuration
    ///
    /// # Example
    /// ```
    /// use matching_networks_finance::training::{EpisodicTrainer, TrainingConfig};
    ///
    /// let config = TrainingConfig::default();
    /// let trainer = EpisodicTrainer::new(config);
    /// ```
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            train_data: HashMap::new(),
            val_data: HashMap::new(),
            rng: rand::thread_rng(),
        }
    }

    /// Add training data
    ///
    /// # Arguments
    /// * `features` - Feature matrix [num_samples, feature_dim]
    /// * `labels` - Label vector [num_samples]
    /// * `val_split` - Fraction of data to use for validation (0.0 to 1.0)
    pub fn add_data(&mut self, features: &Array2<f64>, labels: &Array1<usize>, val_split: f64) {
        // Group by class
        let mut class_data: HashMap<usize, Vec<Array1<f64>>> = HashMap::new();

        for (i, &label) in labels.iter().enumerate() {
            let feature = features.row(i).to_owned();
            class_data.entry(label).or_default().push(feature);
        }

        // Split into train/val
        for (class_id, mut samples) in class_data {
            samples.shuffle(&mut self.rng);
            let val_size = (samples.len() as f64 * val_split) as usize;

            let val_samples: Vec<_> = samples.drain(..val_size).collect();
            let train_samples = samples;

            self.train_data.entry(class_id).or_default().extend(train_samples);
            self.val_data.entry(class_id).or_default().extend(val_samples);
        }
    }

    /// Generate a random episode from the data
    pub fn generate_episode(&mut self, from_validation: bool) -> Option<Episode> {
        let data = if from_validation {
            &self.val_data
        } else {
            &self.train_data
        };

        // Get classes with enough samples
        let available_classes: Vec<usize> = data
            .iter()
            .filter(|(_, samples)| samples.len() >= self.config.k_shot + self.config.n_query)
            .map(|(&class_id, _)| class_id)
            .collect();

        if available_classes.len() < self.config.n_way {
            return None;
        }

        // Sample N classes
        let mut classes: Vec<usize> = available_classes;
        classes.shuffle(&mut self.rng);
        classes.truncate(self.config.n_way);

        // Sample support and query sets
        let mut support_features = Vec::new();
        let mut support_labels = Vec::new();
        let mut query_features = Vec::new();
        let mut query_labels = Vec::new();

        for (new_label, &original_class) in classes.iter().enumerate() {
            let samples = if from_validation {
                &self.val_data[&original_class]
            } else {
                &self.train_data[&original_class]
            };

            // Randomly select samples
            let mut indices: Vec<usize> = (0..samples.len()).collect();
            indices.shuffle(&mut self.rng);

            // Support set
            for &idx in indices.iter().take(self.config.k_shot) {
                support_features.push(samples[idx].clone());
                support_labels.push(new_label);
            }

            // Query set
            for &idx in indices.iter().skip(self.config.k_shot).take(self.config.n_query) {
                query_features.push(samples[idx].clone());
                query_labels.push(new_label);
            }
        }

        // Convert to arrays
        let feature_dim = support_features[0].len();
        let support_flat: Vec<f64> = support_features.into_iter().flatten().collect();
        let query_flat: Vec<f64> = query_features.into_iter().flatten().collect();

        Some(Episode {
            support_features: Array2::from_shape_vec(
                (self.config.n_way * self.config.k_shot, feature_dim),
                support_flat,
            )
            .ok()?,
            support_labels: Array1::from_vec(support_labels),
            query_features: Array2::from_shape_vec(
                (self.config.n_way * self.config.n_query, feature_dim),
                query_flat,
            )
            .ok()?,
            query_labels: Array1::from_vec(query_labels),
            classes,
        })
    }

    /// Train the network using episodic training
    ///
    /// # Arguments
    /// * `network` - The matching network to train
    ///
    /// # Returns
    /// Training metrics
    pub fn train(&mut self, network: &MatchingNetwork) -> TrainingMetrics {
        let mut metrics = TrainingMetrics::default();
        let mut episodes_without_improvement = 0;

        for episode_num in 0..self.config.num_episodes {
            // Generate training episode
            let episode = match self.generate_episode(false) {
                Some(e) => e,
                None => {
                    eprintln!("Warning: Could not generate training episode");
                    continue;
                }
            };

            // Forward pass and compute accuracy
            let predictions = network.predict(
                &episode.support_features,
                &episode.support_labels,
                &episode.query_features,
            );

            let accuracy = self.compute_accuracy(&predictions, &episode.query_labels);
            let loss = self.compute_loss(network, &episode);

            metrics.train_accuracy.push(accuracy);
            metrics.train_loss.push(loss);

            // Validation
            if episode_num % self.config.val_frequency == 0 {
                let val_accuracy = self.validate(network);
                metrics.val_accuracy.push(val_accuracy);

                if val_accuracy > metrics.best_val_accuracy {
                    metrics.best_val_accuracy = val_accuracy;
                    metrics.best_episode = episode_num;
                    episodes_without_improvement = 0;
                } else {
                    episodes_without_improvement += 1;
                }

                // Log progress
                println!(
                    "Episode {}: train_acc={:.4}, val_acc={:.4}, loss={:.4}",
                    episode_num, accuracy, val_accuracy, loss
                );

                // Early stopping
                if episodes_without_improvement >= self.config.patience {
                    println!(
                        "Early stopping at episode {} (no improvement for {} validations)",
                        episode_num, self.config.patience
                    );
                    break;
                }
            }
        }

        metrics
    }

    /// Validate on held-out data
    fn validate(&mut self, network: &MatchingNetwork) -> f64 {
        const NUM_VAL_EPISODES: usize = 10;
        let mut accuracies = Vec::new();

        for _ in 0..NUM_VAL_EPISODES {
            if let Some(episode) = self.generate_episode(true) {
                let predictions = network.predict(
                    &episode.support_features,
                    &episode.support_labels,
                    &episode.query_features,
                );

                let accuracy = self.compute_accuracy(&predictions, &episode.query_labels);
                accuracies.push(accuracy);
            }
        }

        if accuracies.is_empty() {
            0.0
        } else {
            accuracies.iter().sum::<f64>() / accuracies.len() as f64
        }
    }

    /// Compute accuracy
    fn compute_accuracy(&self, predictions: &Array1<usize>, labels: &Array1<usize>) -> f64 {
        let correct: usize = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(&pred, &label)| pred == label)
            .count();

        correct as f64 / labels.len() as f64
    }

    /// Compute cross-entropy loss (for monitoring)
    fn compute_loss(&self, network: &MatchingNetwork, episode: &Episode) -> f64 {
        let probabilities = network.forward(
            &episode.support_features,
            &episode.support_labels,
            &episode.query_features,
        );

        let mut total_loss = 0.0;
        for (i, &label) in episode.query_labels.iter().enumerate() {
            let prob = probabilities[[i, label]].max(1e-10);
            total_loss -= prob.ln();
        }

        total_loss / episode.query_labels.len() as f64
    }

    /// Get the training configuration
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }

    /// Get number of classes in training data
    pub fn num_classes(&self) -> usize {
        self.train_data.len()
    }

    /// Get number of samples per class
    pub fn samples_per_class(&self) -> HashMap<usize, usize> {
        self.train_data
            .iter()
            .map(|(&class_id, samples)| (class_id, samples.len()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn create_synthetic_data() -> (Array2<f64>, Array1<usize>) {
        let num_samples = 500;
        let num_classes = 5;
        let feature_dim = 20;

        let mut features = Vec::new();
        let mut labels = Vec::new();

        let mut rng = rand::thread_rng();

        for class_id in 0..num_classes {
            // Create class center
            let center: Vec<f64> = (0..feature_dim)
                .map(|_| rng.gen_range(-1.0..1.0) + class_id as f64)
                .collect();

            // Generate samples around center
            for _ in 0..num_samples / num_classes {
                let sample: Vec<f64> = center
                    .iter()
                    .map(|&c| c + rng.gen_range(-0.3..0.3))
                    .collect();
                features.extend(sample);
                labels.push(class_id);
            }
        }

        let features = Array2::from_shape_vec(
            (num_samples, feature_dim),
            features,
        )
        .unwrap();
        let labels = Array1::from_vec(labels);

        (features, labels)
    }

    #[test]
    fn test_trainer_creation() {
        let config = TrainingConfig {
            n_way: 3,
            k_shot: 5,
            n_query: 5,
            num_episodes: 100,
            ..Default::default()
        };

        let trainer = EpisodicTrainer::new(config);
        assert_eq!(trainer.config().n_way, 3);
        assert_eq!(trainer.config().k_shot, 5);
    }

    #[test]
    fn test_add_data() {
        let config = TrainingConfig::default();
        let mut trainer = EpisodicTrainer::new(config);

        let (features, labels) = create_synthetic_data();
        trainer.add_data(&features, &labels, 0.2);

        assert_eq!(trainer.num_classes(), 5);

        // Check samples were split
        let samples_per_class = trainer.samples_per_class();
        for (&_class_id, &count) in &samples_per_class {
            assert!(count > 0);
            assert!(count < 100); // Should be ~80% of 100
        }
    }

    #[test]
    fn test_generate_episode() {
        let config = TrainingConfig {
            n_way: 3,
            k_shot: 5,
            n_query: 5,
            ..Default::default()
        };
        let mut trainer = EpisodicTrainer::new(config);

        let (features, labels) = create_synthetic_data();
        trainer.add_data(&features, &labels, 0.2);

        let episode = trainer.generate_episode(false).unwrap();

        assert_eq!(episode.support_features.nrows(), 3 * 5); // n_way * k_shot
        assert_eq!(episode.support_labels.len(), 3 * 5);
        assert_eq!(episode.query_features.nrows(), 3 * 5); // n_way * n_query
        assert_eq!(episode.query_labels.len(), 3 * 5);
        assert_eq!(episode.classes.len(), 3);
    }

    #[test]
    fn test_compute_accuracy() {
        let config = TrainingConfig::default();
        let trainer = EpisodicTrainer::new(config);

        let predictions = Array1::from_vec(vec![0, 1, 2, 0, 1]);
        let labels = Array1::from_vec(vec![0, 1, 2, 1, 1]);

        let accuracy = trainer.compute_accuracy(&predictions, &labels);
        assert!((accuracy - 0.8).abs() < 1e-10); // 4/5 correct
    }
}
