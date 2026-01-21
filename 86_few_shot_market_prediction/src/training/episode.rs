//! Episodic training for few-shot learning

use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for episode generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeConfig {
    /// Number of classes per episode (N-way)
    pub n_way: usize,
    /// Number of support examples per class (K-shot)
    pub k_shot: usize,
    /// Number of query examples per class
    pub n_query: usize,
}

impl Default for EpisodeConfig {
    fn default() -> Self {
        Self {
            n_way: 5,
            k_shot: 5,
            n_query: 10,
        }
    }
}

/// A single training episode containing support and query sets
#[derive(Debug, Clone)]
pub struct Episode {
    /// Support set features
    pub support_features: Array2<f64>,
    /// Support set labels (mapped to 0..n_way)
    pub support_labels: Vec<usize>,
    /// Query set features
    pub query_features: Array2<f64>,
    /// Query set labels (mapped to 0..n_way)
    pub query_labels: Vec<usize>,
    /// Mapping from episode labels to original labels
    pub label_mapping: HashMap<usize, usize>,
}

impl Episode {
    /// Get number of classes in this episode
    pub fn n_way(&self) -> usize {
        self.label_mapping.len()
    }

    /// Get number of support examples per class
    pub fn k_shot(&self) -> usize {
        if self.label_mapping.is_empty() {
            return 0;
        }
        self.support_labels.len() / self.label_mapping.len()
    }

    /// Get number of query examples per class
    pub fn n_query(&self) -> usize {
        if self.label_mapping.is_empty() {
            return 0;
        }
        self.query_labels.len() / self.label_mapping.len()
    }
}

/// Generator for training episodes
pub struct EpisodeGenerator {
    config: EpisodeConfig,
    /// Features grouped by class
    class_features: HashMap<usize, Vec<Vec<f64>>>,
    /// Feature dimension
    feature_dim: usize,
}

impl EpisodeGenerator {
    /// Create a new episode generator from data
    pub fn new(
        features: &Array2<f64>,
        labels: &[usize],
        config: EpisodeConfig,
    ) -> Self {
        let feature_dim = features.ncols();

        // Group features by class
        let mut class_features: HashMap<usize, Vec<Vec<f64>>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            let feature_vec: Vec<f64> = features.row(i).to_vec();
            class_features.entry(label).or_default().push(feature_vec);
        }

        Self {
            config,
            class_features,
            feature_dim,
        }
    }

    /// Generate a single episode
    pub fn generate_episode(&self) -> Option<Episode> {
        let mut rng = rand::thread_rng();

        // Get available classes
        let available_classes: Vec<usize> = self.class_features.keys().cloned().collect();

        if available_classes.len() < self.config.n_way {
            return None;
        }

        // Sample classes for this episode
        let mut sampled_classes = available_classes.clone();
        sampled_classes.shuffle(&mut rng);
        sampled_classes.truncate(self.config.n_way);

        // Create label mapping (episode label -> original label)
        let label_mapping: HashMap<usize, usize> = sampled_classes
            .iter()
            .enumerate()
            .map(|(i, &orig)| (i, orig))
            .collect();

        let mut support_data: Vec<f64> = Vec::new();
        let mut support_labels: Vec<usize> = Vec::new();
        let mut query_data: Vec<f64> = Vec::new();
        let mut query_labels: Vec<usize> = Vec::new();

        for (episode_label, &orig_label) in sampled_classes.iter().enumerate() {
            let class_data = &self.class_features[&orig_label];
            let total_needed = self.config.k_shot + self.config.n_query;

            // Sample indices
            let indices: Vec<usize> = if class_data.len() >= total_needed {
                let mut idx: Vec<usize> = (0..class_data.len()).collect();
                idx.shuffle(&mut rng);
                idx.into_iter().take(total_needed).collect()
            } else {
                // Sample with replacement if not enough data
                (0..total_needed)
                    .map(|_| rng.gen_range(0..class_data.len()))
                    .collect()
            };

            // Split into support and query
            for (i, &idx) in indices.iter().enumerate() {
                if i < self.config.k_shot {
                    support_data.extend(&class_data[idx]);
                    support_labels.push(episode_label);
                } else {
                    query_data.extend(&class_data[idx]);
                    query_labels.push(episode_label);
                }
            }
        }

        let n_support = self.config.n_way * self.config.k_shot;
        let n_query = self.config.n_way * self.config.n_query;

        let support_features = Array2::from_shape_vec(
            (n_support, self.feature_dim),
            support_data,
        ).ok()?;

        let query_features = Array2::from_shape_vec(
            (n_query, self.feature_dim),
            query_data,
        ).ok()?;

        Some(Episode {
            support_features,
            support_labels,
            query_features,
            query_labels,
            label_mapping,
        })
    }

    /// Generate multiple episodes
    pub fn generate_episodes(&self, n_episodes: usize) -> Vec<Episode> {
        (0..n_episodes)
            .filter_map(|_| self.generate_episode())
            .collect()
    }

    /// Get number of available classes
    pub fn num_classes(&self) -> usize {
        self.class_features.len()
    }

    /// Get feature dimension
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> (Array2<f64>, Vec<usize>) {
        let n_classes = 5;
        let n_per_class = 20;
        let dim = 10;

        let mut data = Vec::new();
        let mut labels = Vec::new();

        for class in 0..n_classes {
            for _ in 0..n_per_class {
                for _ in 0..dim {
                    data.push(class as f64 + rand::random::<f64>() * 0.1);
                }
                labels.push(class);
            }
        }

        let features = Array2::from_shape_vec((n_classes * n_per_class, dim), data).unwrap();
        (features, labels)
    }

    #[test]
    fn test_episode_generation() {
        let (features, labels) = create_test_data();
        let config = EpisodeConfig {
            n_way: 3,
            k_shot: 5,
            n_query: 5,
        };

        let generator = EpisodeGenerator::new(&features, &labels, config);
        let episode = generator.generate_episode().unwrap();

        assert_eq!(episode.n_way(), 3);
        assert_eq!(episode.support_features.nrows(), 15); // 3 * 5
        assert_eq!(episode.query_features.nrows(), 15); // 3 * 5
    }

    #[test]
    fn test_multiple_episodes() {
        let (features, labels) = create_test_data();
        let config = EpisodeConfig::default();

        let generator = EpisodeGenerator::new(&features, &labels, config);
        let episodes = generator.generate_episodes(10);

        assert_eq!(episodes.len(), 10);
    }
}
