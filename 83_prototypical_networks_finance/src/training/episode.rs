//! Episodic training data generation for few-shot learning
//!
//! This module implements the episode generation strategy used in
//! prototypical networks for few-shot learning.

use crate::data::MarketRegime;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use std::collections::HashMap;

/// Configuration for episode generation
#[derive(Debug, Clone)]
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
            n_way: 5, // All 5 market regimes
            k_shot: 5, // 5 examples per regime
            n_query: 15, // 15 queries per regime
        }
    }
}

/// A single training episode for few-shot learning
#[derive(Debug, Clone)]
pub struct Episode {
    /// Support set features (n_way * k_shot, feature_dim)
    pub support_features: Array2<f64>,
    /// Support set labels
    pub support_labels: Vec<usize>,
    /// Query set features (n_way * n_query, feature_dim)
    pub query_features: Array2<f64>,
    /// Query set labels (ground truth for evaluation)
    pub query_labels: Vec<usize>,
    /// Classes included in this episode
    pub classes: Vec<MarketRegime>,
}

impl Episode {
    /// Get the number of classes in this episode
    pub fn n_way(&self) -> usize {
        self.classes.len()
    }

    /// Get the number of support examples per class
    pub fn k_shot(&self) -> usize {
        self.support_labels.len() / self.classes.len()
    }

    /// Get the number of query examples per class
    pub fn n_query(&self) -> usize {
        self.query_labels.len() / self.classes.len()
    }

    /// Get support set for a specific class
    pub fn support_for_class(&self, class_idx: usize) -> Array2<f64> {
        let k = self.k_shot();
        let start = class_idx * k;
        let end = start + k;
        self.support_features.slice(ndarray::s![start..end, ..]).to_owned()
    }
}

/// Generator for training episodes
pub struct EpisodeGenerator {
    config: EpisodeConfig,
    /// Data organized by class: HashMap<class_index, Vec<feature_vectors>>
    class_data: HashMap<usize, Vec<Array1<f64>>>,
    rng: StdRng,
}

impl EpisodeGenerator {
    /// Create a new episode generator
    pub fn new(config: EpisodeConfig) -> Self {
        Self {
            config,
            class_data: HashMap::new(),
            rng: StdRng::from_entropy(),
        }
    }

    /// Create a new episode generator with a fixed seed for reproducibility
    pub fn with_seed(config: EpisodeConfig, seed: u64) -> Self {
        Self {
            config,
            class_data: HashMap::new(),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Add data for a specific class
    pub fn add_class_data(&mut self, class_idx: usize, data: Vec<Array1<f64>>) {
        self.class_data
            .entry(class_idx)
            .or_insert_with(Vec::new)
            .extend(data);
    }

    /// Add labeled data points
    pub fn add_data(&mut self, features: Array2<f64>, labels: &[usize]) {
        for (i, &label) in labels.iter().enumerate() {
            let feature = features.row(i).to_owned();
            self.class_data
                .entry(label)
                .or_insert_with(Vec::new)
                .push(feature);
        }
    }

    /// Check if we have enough data for episode generation
    pub fn can_generate(&self) -> bool {
        let min_samples = self.config.k_shot + self.config.n_query;
        self.class_data.len() >= self.config.n_way
            && self.class_data.values().all(|v| v.len() >= min_samples)
    }

    /// Get the number of available classes
    pub fn num_classes(&self) -> usize {
        self.class_data.len()
    }

    /// Get the number of samples for each class
    pub fn samples_per_class(&self) -> HashMap<usize, usize> {
        self.class_data
            .iter()
            .map(|(&k, v)| (k, v.len()))
            .collect()
    }

    /// Generate a single episode
    pub fn generate_episode(&mut self) -> Option<Episode> {
        if !self.can_generate() {
            return None;
        }

        // Select n_way classes randomly
        let available_classes: Vec<usize> = self.class_data.keys().cloned().collect();
        let selected_classes: Vec<usize> = available_classes
            .choose_multiple(&mut self.rng, self.config.n_way)
            .cloned()
            .collect();

        let regimes: Vec<MarketRegime> = selected_classes
            .iter()
            .filter_map(|&idx| MarketRegime::from_index(idx))
            .collect();

        let feature_dim = self.class_data.values().next()?.first()?.len();
        let total_support = self.config.n_way * self.config.k_shot;
        let total_query = self.config.n_way * self.config.n_query;

        let mut support_features = Array2::zeros((total_support, feature_dim));
        let mut support_labels = Vec::with_capacity(total_support);
        let mut query_features = Array2::zeros((total_query, feature_dim));
        let mut query_labels = Vec::with_capacity(total_query);

        for (episode_class_idx, &original_class) in selected_classes.iter().enumerate() {
            let class_samples = self.class_data.get(&original_class)?;

            // Sample k_shot + n_query samples without replacement
            let total_needed = self.config.k_shot + self.config.n_query;
            let indices: Vec<usize> = (0..class_samples.len())
                .collect::<Vec<_>>()
                .choose_multiple(&mut self.rng, total_needed)
                .cloned()
                .collect();

            // Split into support and query
            for (i, &idx) in indices.iter().enumerate() {
                if i < self.config.k_shot {
                    // Support set
                    let row_idx = episode_class_idx * self.config.k_shot + i;
                    support_features
                        .row_mut(row_idx)
                        .assign(&class_samples[idx]);
                    support_labels.push(episode_class_idx);
                } else {
                    // Query set
                    let query_idx = i - self.config.k_shot;
                    let row_idx = episode_class_idx * self.config.n_query + query_idx;
                    query_features
                        .row_mut(row_idx)
                        .assign(&class_samples[idx]);
                    query_labels.push(episode_class_idx);
                }
            }
        }

        Some(Episode {
            support_features,
            support_labels,
            query_features,
            query_labels,
            classes: regimes,
        })
    }

    /// Generate multiple episodes
    pub fn generate_episodes(&mut self, n_episodes: usize) -> Vec<Episode> {
        (0..n_episodes)
            .filter_map(|_| self.generate_episode())
            .collect()
    }

    /// Generate an iterator over episodes
    pub fn episode_iter(&mut self, n_episodes: usize) -> EpisodeIterator<'_> {
        EpisodeIterator {
            generator: self,
            remaining: n_episodes,
        }
    }
}

/// Iterator over episodes
pub struct EpisodeIterator<'a> {
    generator: &'a mut EpisodeGenerator,
    remaining: usize,
}

impl<'a> Iterator for EpisodeIterator<'a> {
    type Item = Episode;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        self.generator.generate_episode()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> HashMap<usize, Vec<Array1<f64>>> {
        let mut data = HashMap::new();
        let mut rng = StdRng::seed_from_u64(42);

        for class in 0..5 {
            let samples: Vec<Array1<f64>> = (0..30)
                .map(|_| {
                    let base = class as f64;
                    Array1::from_vec(vec![
                        base + rng.gen::<f64>() * 0.5,
                        base * 2.0 + rng.gen::<f64>() * 0.5,
                        base * 0.5 + rng.gen::<f64>() * 0.5,
                    ])
                })
                .collect();
            data.insert(class, samples);
        }

        data
    }

    #[test]
    fn test_episode_generation() {
        let config = EpisodeConfig {
            n_way: 3,
            k_shot: 5,
            n_query: 10,
        };

        let mut generator = EpisodeGenerator::with_seed(config.clone(), 42);

        for (class, samples) in create_test_data() {
            generator.add_class_data(class, samples);
        }

        assert!(generator.can_generate());

        let episode = generator.generate_episode().unwrap();

        assert_eq!(episode.n_way(), 3);
        assert_eq!(episode.k_shot(), 5);
        assert_eq!(episode.n_query(), 10);
        assert_eq!(episode.support_features.nrows(), 15); // 3 * 5
        assert_eq!(episode.query_features.nrows(), 30); // 3 * 10
    }

    #[test]
    fn test_episode_iterator() {
        let config = EpisodeConfig::default();
        let mut generator = EpisodeGenerator::with_seed(config, 42);

        for (class, samples) in create_test_data() {
            generator.add_class_data(class, samples);
        }

        let episodes: Vec<_> = generator.episode_iter(10).collect();
        assert_eq!(episodes.len(), 10);
    }

    #[test]
    fn test_insufficient_data() {
        let config = EpisodeConfig {
            n_way: 5,
            k_shot: 50, // Too many samples required
            n_query: 10,
        };

        let mut generator = EpisodeGenerator::new(config);

        for (class, samples) in create_test_data() {
            generator.add_class_data(class, samples);
        }

        assert!(!generator.can_generate());
    }
}
