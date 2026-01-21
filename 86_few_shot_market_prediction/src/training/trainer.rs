//! Meta-training infrastructure

use super::episode::{Episode, EpisodeConfig, EpisodeGenerator};
use crate::methods::{FewShotConfig, FewShotLearner, FewShotPredictor, PredictionResult};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Configuration for meta-trainer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaTrainerConfig {
    /// Episode configuration
    pub episode_config: EpisodeConfig,
    /// Number of training episodes
    pub n_episodes: usize,
    /// Evaluation frequency (episodes)
    pub eval_frequency: usize,
    /// Number of evaluation episodes
    pub n_eval_episodes: usize,
}

impl Default for MetaTrainerConfig {
    fn default() -> Self {
        Self {
            episode_config: EpisodeConfig::default(),
            n_episodes: 1000,
            eval_frequency: 100,
            n_eval_episodes: 50,
        }
    }
}

/// Training result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Episode number
    pub episode: usize,
    /// Accuracy on this episode
    pub accuracy: f64,
    /// Average confidence
    pub avg_confidence: f64,
}

/// Meta-trainer for few-shot learning
pub struct MetaTrainer {
    config: MetaTrainerConfig,
    fs_config: FewShotConfig,
    training_history: Vec<TrainingResult>,
}

impl MetaTrainer {
    /// Create a new meta-trainer
    pub fn new(config: MetaTrainerConfig, fs_config: FewShotConfig) -> Self {
        Self {
            config,
            fs_config,
            training_history: Vec::new(),
        }
    }

    /// Train on generated episodes (demonstration)
    ///
    /// Note: This is a simplified demonstration. In practice, you would
    /// update the network parameters during training. This implementation
    /// shows the episodic training loop structure.
    pub fn train(&mut self, features: &Array2<f64>, labels: &[usize]) -> Vec<TrainingResult> {
        let generator = EpisodeGenerator::new(
            features,
            labels,
            self.config.episode_config.clone(),
        );

        let mut results = Vec::new();

        for episode_idx in 0..self.config.n_episodes {
            if let Some(episode) = generator.generate_episode() {
                let result = self.train_episode(&episode, episode_idx);
                results.push(result.clone());
                self.training_history.push(result);
            }
        }

        results
    }

    /// Train on a single episode
    fn train_episode(&self, episode: &Episode, episode_idx: usize) -> TrainingResult {
        // Create a predictor for this episode
        let mut predictor = FewShotPredictor::new(self.fs_config.clone());

        // Fit on support set
        predictor.fit(&episode.support_features, &episode.support_labels);

        // Evaluate on query set
        let predictions = predictor.predict(&episode.query_features);

        // Calculate metrics
        let (accuracy, avg_confidence) = self.calculate_metrics(&predictions, &episode.query_labels);

        TrainingResult {
            episode: episode_idx,
            accuracy,
            avg_confidence,
        }
    }

    /// Calculate accuracy and average confidence
    fn calculate_metrics(
        &self,
        predictions: &[PredictionResult],
        true_labels: &[usize],
    ) -> (f64, f64) {
        if predictions.is_empty() {
            return (0.0, 0.0);
        }

        let correct: usize = predictions
            .iter()
            .zip(true_labels.iter())
            .filter(|(pred, &true_label)| pred.predicted_class == true_label)
            .count();

        let accuracy = correct as f64 / predictions.len() as f64;
        let avg_confidence = predictions.iter().map(|p| p.confidence).sum::<f64>() / predictions.len() as f64;

        (accuracy, avg_confidence)
    }

    /// Get training history
    pub fn history(&self) -> &[TrainingResult] {
        &self.training_history
    }

    /// Get average accuracy over recent episodes
    pub fn recent_accuracy(&self, n_recent: usize) -> f64 {
        if self.training_history.is_empty() {
            return 0.0;
        }

        let recent: Vec<_> = self.training_history.iter().rev().take(n_recent).collect();
        recent.iter().map(|r| r.accuracy).sum::<f64>() / recent.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::FewShotMethod;

    fn create_test_data() -> (Array2<f64>, Vec<usize>) {
        let n_classes = 5;
        let n_per_class = 30;
        let dim = 20;

        let mut data = Vec::new();
        let mut labels = Vec::new();

        for class in 0..n_classes {
            for _ in 0..n_per_class {
                for _ in 0..dim {
                    data.push(class as f64 * 0.5 + rand::random::<f64>() * 0.1);
                }
                labels.push(class);
            }
        }

        let features = Array2::from_shape_vec((n_classes * n_per_class, dim), data).unwrap();
        (features, labels)
    }

    #[test]
    fn test_meta_trainer() {
        let trainer_config = MetaTrainerConfig {
            episode_config: EpisodeConfig {
                n_way: 3,
                k_shot: 5,
                n_query: 5,
            },
            n_episodes: 5,
            ..Default::default()
        };

        let fs_config = FewShotConfig::default()
            .with_method(FewShotMethod::Metric)
            .with_input_dim(20);

        let mut trainer = MetaTrainer::new(trainer_config, fs_config);
        let (features, labels) = create_test_data();

        let results = trainer.train(&features, &labels);
        assert_eq!(results.len(), 5);

        for result in results {
            assert!(result.accuracy >= 0.0 && result.accuracy <= 1.0);
            assert!(result.avg_confidence >= 0.0 && result.avg_confidence <= 1.0);
        }
    }
}
