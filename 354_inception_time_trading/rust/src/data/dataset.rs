//! Dataset loading and management for training
//!
//! This module provides data loading utilities for creating
//! training, validation, and test datasets.

use anyhow::Result;
use ndarray::{Array2, Array3};
use rand::seq::SliceRandom;
use rand::thread_rng;

use super::features::{generate_labels, FeatureBuilder, NormalizationParams};
use super::ohlcv::OHLCVDataset;

/// Trading dataset with windowed features and labels
#[derive(Debug)]
pub struct TradingDataset {
    /// Features as 3D tensor: (samples, window_size, num_features)
    pub features: Array3<f64>,
    /// Labels as 1D vector
    pub labels: Vec<i64>,
    /// Normalization parameters
    pub norm_params: NormalizationParams,
}

impl TradingDataset {
    /// Create dataset from OHLCV data
    pub fn from_ohlcv(
        dataset: &OHLCVDataset,
        window_size: usize,
        stride: usize,
        prediction_horizon: usize,
        threshold_pct: f64,
        feature_names: Vec<String>,
    ) -> Result<Self> {
        let builder = FeatureBuilder::new(window_size, feature_names);
        let (features_2d, norm_params) = builder.build(dataset)?;

        // Generate labels
        let closes = dataset.closes();
        let labels_full = generate_labels(&closes, prediction_horizon, threshold_pct);

        // Create windowed samples
        let num_features = features_2d.ncols();
        let max_start = features_2d.nrows().saturating_sub(window_size);
        let num_samples = (max_start / stride) + 1;

        let mut features = Array3::zeros((num_samples, window_size, num_features));
        let mut labels = Vec::with_capacity(num_samples);

        for (sample_idx, start) in (0..=max_start).step_by(stride).enumerate() {
            let end = start + window_size;

            // Copy window of features
            for t in 0..window_size {
                for f in 0..num_features {
                    features[[sample_idx, t, f]] = features_2d[[start + t, f]];
                }
            }

            // Get label for the end of the window
            let label_idx = (closes.len() - features_2d.nrows()) + end - 1;
            labels.push(labels_full.get(label_idx).copied().unwrap_or(1));
        }

        Ok(Self {
            features,
            labels,
            norm_params,
        })
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Get a batch of samples
    pub fn get_batch(&self, indices: &[usize]) -> (Array3<f64>, Vec<i64>) {
        let batch_size = indices.len();
        let window_size = self.features.dim().1;
        let num_features = self.features.dim().2;

        let mut batch_features = Array3::zeros((batch_size, window_size, num_features));
        let mut batch_labels = Vec::with_capacity(batch_size);

        for (batch_idx, &sample_idx) in indices.iter().enumerate() {
            for t in 0..window_size {
                for f in 0..num_features {
                    batch_features[[batch_idx, t, f]] = self.features[[sample_idx, t, f]];
                }
            }
            batch_labels.push(self.labels[sample_idx]);
        }

        (batch_features, batch_labels)
    }

    /// Shuffle the dataset
    pub fn shuffle(&mut self) {
        let mut indices: Vec<usize> = (0..self.len()).collect();
        indices.shuffle(&mut thread_rng());

        let (shuffled_features, shuffled_labels) = self.get_batch(&indices);
        self.features = shuffled_features;
        self.labels = shuffled_labels;
    }

    /// Get class distribution
    pub fn class_distribution(&self) -> [usize; 3] {
        let mut dist = [0usize; 3];
        for &label in &self.labels {
            if label >= 0 && label < 3 {
                dist[label as usize] += 1;
            }
        }
        dist
    }

    /// Calculate class weights for balanced training
    pub fn class_weights(&self) -> [f64; 3] {
        let dist = self.class_distribution();
        let total = dist.iter().sum::<usize>() as f64;

        let mut weights = [1.0; 3];
        for (i, &count) in dist.iter().enumerate() {
            if count > 0 {
                weights[i] = total / (3.0 * count as f64);
            }
        }
        weights
    }
}

/// Data loader for batched iteration
pub struct DataLoader {
    indices: Vec<usize>,
    batch_size: usize,
    current_pos: usize,
    shuffle: bool,
}

impl DataLoader {
    /// Create a new data loader
    pub fn new(dataset_size: usize, batch_size: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..dataset_size).collect();
        if shuffle {
            indices.shuffle(&mut thread_rng());
        }

        Self {
            indices,
            batch_size,
            current_pos: 0,
            shuffle,
        }
    }

    /// Get number of batches
    pub fn num_batches(&self) -> usize {
        (self.indices.len() + self.batch_size - 1) / self.batch_size
    }

    /// Reset the loader for a new epoch
    pub fn reset(&mut self) {
        self.current_pos = 0;
        if self.shuffle {
            self.indices.shuffle(&mut thread_rng());
        }
    }

    /// Get next batch of indices
    pub fn next_batch(&mut self) -> Option<Vec<usize>> {
        if self.current_pos >= self.indices.len() {
            return None;
        }

        let end = (self.current_pos + self.batch_size).min(self.indices.len());
        let batch = self.indices[self.current_pos..end].to_vec();
        self.current_pos = end;

        Some(batch)
    }
}

impl Iterator for DataLoader {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_loader() {
        let loader = DataLoader::new(100, 32, false);
        assert_eq!(loader.num_batches(), 4); // 32 + 32 + 32 + 4 = 100
    }

    #[test]
    fn test_data_loader_iteration() {
        let mut loader = DataLoader::new(10, 3, false);
        let batches: Vec<_> = loader.by_ref().collect();

        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[3].len(), 1);
    }
}
