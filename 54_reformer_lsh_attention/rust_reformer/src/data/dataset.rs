//! Dataset structures for model training and inference
//!
//! Provides dataset types and utilities for sequence data handling.

use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

/// A single sample for training/inference
#[derive(Debug, Clone)]
pub struct Sample {
    /// Input features [seq_len, n_features]
    pub features: Array2<f64>,
    /// Target value(s) (future returns)
    pub target: Vec<f64>,
    /// Timestamp of the sample
    pub timestamp: u64,
}

/// Dataset containing multiple samples
#[derive(Debug, Clone)]
pub struct Dataset {
    /// All samples
    pub samples: Vec<Sample>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Sequence length
    pub seq_len: usize,
    /// Prediction horizon
    pub horizon: usize,
}

impl Dataset {
    /// Create a new empty dataset
    pub fn new(feature_names: Vec<String>, seq_len: usize, horizon: usize) -> Self {
        Self {
            samples: Vec::new(),
            feature_names,
            seq_len,
            horizon,
        }
    }

    /// Number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Number of features
    pub fn n_features(&self) -> usize {
        self.feature_names.len()
    }

    /// Add a sample
    pub fn push(&mut self, sample: Sample) {
        self.samples.push(sample);
    }

    /// Get a batch of samples as 3D array [batch, seq_len, features]
    pub fn get_batch(&self, indices: &[usize]) -> Option<(Array3<f64>, Array2<f64>)> {
        if indices.is_empty() {
            return None;
        }

        let batch_size = indices.len();
        let n_features = self.n_features();

        let mut features = Array3::zeros((batch_size, self.seq_len, n_features));
        let mut targets = Array2::zeros((batch_size, self.samples[0].target.len()));

        for (batch_idx, &sample_idx) in indices.iter().enumerate() {
            if sample_idx >= self.samples.len() {
                return None;
            }

            let sample = &self.samples[sample_idx];

            // Copy features
            for t in 0..self.seq_len {
                for f in 0..n_features {
                    features[[batch_idx, t, f]] = sample.features[[t, f]];
                }
            }

            // Copy targets
            for (t_idx, &target) in sample.target.iter().enumerate() {
                targets[[batch_idx, t_idx]] = target;
            }
        }

        Some((features, targets))
    }

    /// Split dataset into train and validation sets
    pub fn train_val_split(&self, train_ratio: f64) -> (Dataset, Dataset) {
        let split_idx = (self.samples.len() as f64 * train_ratio) as usize;

        let train_samples = self.samples[..split_idx].to_vec();
        let val_samples = self.samples[split_idx..].to_vec();

        let train = Dataset {
            samples: train_samples,
            feature_names: self.feature_names.clone(),
            seq_len: self.seq_len,
            horizon: self.horizon,
        };

        let val = Dataset {
            samples: val_samples,
            feature_names: self.feature_names.clone(),
            seq_len: self.seq_len,
            horizon: self.horizon,
        };

        (train, val)
    }

    /// Get all features as a 3D array [n_samples, seq_len, n_features]
    pub fn all_features(&self) -> Array3<f64> {
        let n = self.samples.len();
        let n_features = self.n_features();

        let mut features = Array3::zeros((n, self.seq_len, n_features));

        for (i, sample) in self.samples.iter().enumerate() {
            for t in 0..self.seq_len {
                for f in 0..n_features {
                    features[[i, t, f]] = sample.features[[t, f]];
                }
            }
        }

        features
    }

    /// Get all targets as a 2D array [n_samples, horizon]
    pub fn all_targets(&self) -> Array2<f64> {
        let n = self.samples.len();
        if n == 0 {
            return Array2::zeros((0, 0));
        }

        let target_len = self.samples[0].target.len();
        let mut targets = Array2::zeros((n, target_len));

        for (i, sample) in self.samples.iter().enumerate() {
            for (j, &t) in sample.target.iter().enumerate() {
                targets[[i, j]] = t;
            }
        }

        targets
    }

    /// Compute dataset statistics
    pub fn statistics(&self) -> DatasetStatistics {
        let all_targets = self.all_targets();

        let target_mean = all_targets.mean().unwrap_or(0.0);
        let target_std = {
            let variance = all_targets
                .iter()
                .map(|x| (x - target_mean).powi(2))
                .sum::<f64>()
                / (all_targets.len() as f64 - 1.0).max(1.0);
            variance.sqrt()
        };

        DatasetStatistics {
            n_samples: self.samples.len(),
            seq_len: self.seq_len,
            n_features: self.n_features(),
            horizon: self.horizon,
            target_mean,
            target_std,
        }
    }
}

/// Dataset statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStatistics {
    pub n_samples: usize,
    pub seq_len: usize,
    pub n_features: usize,
    pub horizon: usize,
    pub target_mean: f64,
    pub target_std: f64,
}

impl std::fmt::Display for DatasetStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Dataset Statistics:")?;
        writeln!(f, "  Samples: {}", self.n_samples)?;
        writeln!(f, "  Sequence Length: {}", self.seq_len)?;
        writeln!(f, "  Features: {}", self.n_features)?;
        writeln!(f, "  Prediction Horizon: {}", self.horizon)?;
        writeln!(f, "  Target Mean: {:.6}", self.target_mean)?;
        writeln!(f, "  Target Std: {:.6}", self.target_std)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn create_test_sample(seq_len: usize, n_features: usize) -> Sample {
        Sample {
            features: Array2::from_shape_fn((seq_len, n_features), |(i, j)| {
                (i * n_features + j) as f64
            }),
            target: vec![0.01, 0.02],
            timestamp: 1704067200000,
        }
    }

    #[test]
    fn test_dataset_creation() {
        let feature_names = vec!["f1".to_string(), "f2".to_string(), "f3".to_string()];
        let dataset = Dataset::new(feature_names.clone(), 10, 2);

        assert!(dataset.is_empty());
        assert_eq!(dataset.n_features(), 3);
        assert_eq!(dataset.seq_len, 10);
        assert_eq!(dataset.horizon, 2);
    }

    #[test]
    fn test_dataset_push() {
        let feature_names = vec!["f1".to_string(), "f2".to_string(), "f3".to_string()];
        let mut dataset = Dataset::new(feature_names, 10, 2);

        dataset.push(create_test_sample(10, 3));
        dataset.push(create_test_sample(10, 3));

        assert_eq!(dataset.len(), 2);
    }

    #[test]
    fn test_get_batch() {
        let feature_names = vec!["f1".to_string(), "f2".to_string()];
        let mut dataset = Dataset::new(feature_names, 5, 2);

        for _ in 0..10 {
            dataset.push(create_test_sample(5, 2));
        }

        let (features, targets) = dataset.get_batch(&[0, 1, 2]).unwrap();

        assert_eq!(features.dim(), (3, 5, 2));
        assert_eq!(targets.dim(), (3, 2));
    }

    #[test]
    fn test_train_val_split() {
        let feature_names = vec!["f1".to_string()];
        let mut dataset = Dataset::new(feature_names, 5, 1);

        for _ in 0..100 {
            dataset.push(create_test_sample(5, 1));
        }

        let (train, val) = dataset.train_val_split(0.8);

        assert_eq!(train.len(), 80);
        assert_eq!(val.len(), 20);
    }

    #[test]
    fn test_statistics() {
        let feature_names = vec!["f1".to_string()];
        let mut dataset = Dataset::new(feature_names, 5, 2);

        for _ in 0..10 {
            dataset.push(create_test_sample(5, 1));
        }

        let stats = dataset.statistics();

        assert_eq!(stats.n_samples, 10);
        assert_eq!(stats.seq_len, 5);
        assert_eq!(stats.n_features, 1);
    }
}
