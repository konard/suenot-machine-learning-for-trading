//! Dataset handling for training and inference

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, Array3, Axis};
use rand::seq::SliceRandom;

/// Dataset for training ConvNeXt
pub struct Dataset {
    /// Feature data [num_samples, sequence_length, num_features]
    pub features: Array3<f64>,
    /// Labels [num_samples]
    pub labels: Array1<usize>,
    /// Sequence length
    pub seq_length: usize,
    /// Number of features
    pub num_features: usize,
}

impl Dataset {
    /// Create dataset from feature matrix
    ///
    /// # Arguments
    /// * `features` - Feature matrix [num_candles, num_features]
    /// * `seq_length` - Sequence length for each sample
    pub fn from_features(features: Array2<f64>, seq_length: usize) -> Result<Self> {
        let (num_candles, num_features) = features.dim();

        if num_candles < seq_length + 1 {
            return Err(anyhow!(
                "Not enough data: {} candles, need at least {}",
                num_candles,
                seq_length + 1
            ));
        }

        let num_samples = num_candles - seq_length;

        // Create sequences
        let mut data = Array3::zeros((num_samples, seq_length, num_features));
        let mut labels = Array1::zeros(num_samples);

        for i in 0..num_samples {
            // Copy sequence
            for j in 0..seq_length {
                for k in 0..num_features {
                    data[[i, j, k]] = features[[i + j, k]];
                }
            }

            // Label based on next candle's return
            // 0 = Long (price goes up), 1 = Short (price goes down), 2 = Hold
            let current_close = features[[i + seq_length - 1, 3]]; // close is feature 3
            let next_close = features[[i + seq_length, 3]];
            let return_pct = (next_close - current_close) / current_close.abs().max(1e-8);

            labels[i] = if return_pct > 0.001 {
                0 // Long
            } else if return_pct < -0.001 {
                1 // Short
            } else {
                2 // Hold
            };
        }

        Ok(Self {
            features: data,
            labels,
            seq_length,
            num_features,
        })
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.features.dim().0
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a single sample
    pub fn get(&self, idx: usize) -> Option<(Array2<f64>, usize)> {
        if idx >= self.len() {
            return None;
        }

        let x = self.features.index_axis(Axis(0), idx).to_owned();
        let y = self.labels[idx];

        Some((x, y))
    }

    /// Split into train and test sets
    pub fn train_test_split(&self, test_ratio: f64) -> (Dataset, Dataset) {
        let n = self.len();
        let test_size = (n as f64 * test_ratio) as usize;
        let train_size = n - test_size;

        let train_features = self.features.slice(ndarray::s![..train_size, .., ..]).to_owned();
        let train_labels = self.labels.slice(ndarray::s![..train_size]).to_owned();

        let test_features = self.features.slice(ndarray::s![train_size.., .., ..]).to_owned();
        let test_labels = self.labels.slice(ndarray::s![train_size..]).to_owned();

        let train = Dataset {
            features: train_features,
            labels: train_labels,
            seq_length: self.seq_length,
            num_features: self.num_features,
        };

        let test = Dataset {
            features: test_features,
            labels: test_labels,
            seq_length: self.seq_length,
            num_features: self.num_features,
        };

        (train, test)
    }

    /// Generate batches for training
    ///
    /// Returns iterator over (features, labels) batches
    /// Features shape: [batch_size, num_features, seq_length] (transposed for Conv1d)
    pub fn batches(&self, batch_size: usize) -> BatchIterator {
        let mut indices: Vec<usize> = (0..self.len()).collect();
        indices.shuffle(&mut rand::thread_rng());

        BatchIterator {
            dataset: self,
            indices,
            batch_size,
            current: 0,
        }
    }

    /// Get class distribution
    pub fn class_distribution(&self) -> [usize; 3] {
        let mut counts = [0usize; 3];
        for &label in self.labels.iter() {
            if label < 3 {
                counts[label] += 1;
            }
        }
        counts
    }

    /// Calculate class weights for balanced training
    pub fn class_weights(&self) -> [f64; 3] {
        let counts = self.class_distribution();
        let total = counts.iter().sum::<usize>() as f64;
        let n_classes = 3.0;

        [
            total / (n_classes * counts[0].max(1) as f64),
            total / (n_classes * counts[1].max(1) as f64),
            total / (n_classes * counts[2].max(1) as f64),
        ]
    }
}

/// Iterator over batches
pub struct BatchIterator<'a> {
    dataset: &'a Dataset,
    indices: Vec<usize>,
    batch_size: usize,
    current: usize,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = (Array3<f64>, Array1<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.indices.len() {
            return None;
        }

        let end = (self.current + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current..end];
        let actual_batch_size = batch_indices.len();

        // Allocate batch tensors
        // Transpose to [batch, features, seq_length] for Conv1d
        let mut features = Array3::zeros((
            actual_batch_size,
            self.dataset.num_features,
            self.dataset.seq_length,
        ));
        let mut labels = Array1::zeros(actual_batch_size);

        for (batch_idx, &data_idx) in batch_indices.iter().enumerate() {
            // Transpose from [seq, features] to [features, seq]
            for f in 0..self.dataset.num_features {
                for s in 0..self.dataset.seq_length {
                    features[[batch_idx, f, s]] = self.dataset.features[[data_idx, s, f]];
                }
            }
            labels[batch_idx] = self.dataset.labels[data_idx];
        }

        self.current = end;

        Some((features, labels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_features(n: usize, num_features: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, num_features), |(i, j)| {
            (i as f64 * 0.01) + (j as f64 * 0.001)
        })
    }

    #[test]
    fn test_dataset_creation() {
        let features = create_test_features(300, 20);
        let dataset = Dataset::from_features(features, 256).unwrap();

        assert_eq!(dataset.len(), 300 - 256);
        assert_eq!(dataset.seq_length, 256);
        assert_eq!(dataset.num_features, 20);
    }

    #[test]
    fn test_train_test_split() {
        let features = create_test_features(500, 20);
        let dataset = Dataset::from_features(features, 100).unwrap();

        let (train, test) = dataset.train_test_split(0.2);

        assert!(train.len() > test.len());
        assert_eq!(train.len() + test.len(), dataset.len());
    }

    #[test]
    fn test_batches() {
        let features = create_test_features(500, 20);
        let dataset = Dataset::from_features(features, 100).unwrap();

        let batch_size = 32;
        let mut total_samples = 0;

        for (x, y) in dataset.batches(batch_size) {
            assert!(x.dim().0 <= batch_size);
            assert_eq!(x.dim().0, y.len());
            assert_eq!(x.dim().1, 20); // num_features
            assert_eq!(x.dim().2, 100); // seq_length
            total_samples += x.dim().0;
        }

        assert_eq!(total_samples, dataset.len());
    }

    #[test]
    fn test_class_distribution() {
        let features = create_test_features(500, 20);
        let dataset = Dataset::from_features(features, 100).unwrap();

        let dist = dataset.class_distribution();
        assert_eq!(dist.iter().sum::<usize>(), dataset.len());
    }
}
