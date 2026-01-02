//! Dataset creation and management

use super::features::Features;
use crate::api::Candle;
use ndarray::{Array2, Array3};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

/// A single training sample
#[derive(Debug, Clone)]
pub struct Sample {
    /// Input features [channels, sequence_length]
    pub features: Array2<f32>,
    /// Target label
    pub label: u8,
    /// Timestamp of the sample
    pub timestamp: i64,
}

/// Dataset for time series classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    /// All samples features [num_samples, channels, sequence_length]
    pub x: Array3<f32>,
    /// All labels
    pub y: Vec<u8>,
    /// Timestamps for each sample
    pub timestamps: Vec<i64>,
    /// Sequence length
    pub sequence_length: usize,
    /// Number of features (channels)
    pub num_features: usize,
    /// Forward window for label generation
    pub forward_window: usize,
    /// Threshold for classification
    pub threshold: f32,
}

impl Dataset {
    /// Create a dataset from candle data
    ///
    /// # Arguments
    ///
    /// * `candles` - Vector of OHLCV candles
    /// * `sequence_length` - Number of time steps per sample
    /// * `forward_window` - Number of steps ahead for label generation
    /// * `threshold` - Return threshold for classification (e.g., 0.002 = 0.2%)
    pub fn from_candles(
        candles: Vec<Candle>,
        sequence_length: usize,
        forward_window: usize,
        threshold: f32,
    ) -> anyhow::Result<Self> {
        let n = candles.len();

        if n < sequence_length + forward_window {
            anyhow::bail!(
                "Not enough candles. Need at least {} but got {}",
                sequence_length + forward_window,
                n
            );
        }

        // Generate features
        let feature_gen = Features::default();
        let all_features = feature_gen.generate(&candles);
        let num_features = all_features.shape()[0];

        // Generate labels
        let closes: Vec<f32> = candles.iter().map(|c| c.close as f32).collect();
        let labels = feature_gen.generate_labels(&closes, forward_window, threshold);

        // Create sliding window samples
        let num_samples = n - sequence_length - forward_window + 1;
        let mut x = Array3::zeros((num_samples, num_features, sequence_length));
        let mut y = Vec::with_capacity(num_samples);
        let mut timestamps = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            // Extract sequence
            for f in 0..num_features {
                for t in 0..sequence_length {
                    x[[i, f, t]] = all_features[[f, i + t]];
                }
            }

            // Get label at the end of the sequence
            y.push(labels[i + sequence_length - 1]);
            timestamps.push(candles[i + sequence_length - 1].timestamp);
        }

        Ok(Self {
            x,
            y,
            timestamps,
            sequence_length,
            num_features,
            forward_window,
            threshold,
        })
    }

    /// Get the number of samples
    pub fn len(&self) -> usize {
        self.y.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.y.is_empty()
    }

    /// Get a single sample
    pub fn get(&self, idx: usize) -> Option<Sample> {
        if idx >= self.len() {
            return None;
        }

        let features = self.x.slice(ndarray::s![idx, .., ..]).to_owned();

        Some(Sample {
            features,
            label: self.y[idx],
            timestamp: self.timestamps[idx],
        })
    }

    /// Get a batch of samples
    pub fn get_batch(&self, indices: &[usize]) -> (Array3<f32>, Vec<u8>) {
        let batch_size = indices.len();
        let mut batch_x = Array3::zeros((batch_size, self.num_features, self.sequence_length));
        let mut batch_y = Vec::with_capacity(batch_size);

        for (b, &idx) in indices.iter().enumerate() {
            for f in 0..self.num_features {
                for t in 0..self.sequence_length {
                    batch_x[[b, f, t]] = self.x[[idx, f, t]];
                }
            }
            batch_y.push(self.y[idx]);
        }

        (batch_x, batch_y)
    }

    /// Split dataset into train/validation/test sets
    ///
    /// # Arguments
    ///
    /// * `train_ratio` - Ratio for training set (e.g., 0.7)
    /// * `val_ratio` - Ratio for validation set (e.g., 0.15)
    ///
    /// The remaining goes to test set
    pub fn split(
        &self,
        train_ratio: f32,
        val_ratio: f32,
    ) -> (Dataset, Dataset, Dataset) {
        let n = self.len();
        let train_end = (n as f32 * train_ratio) as usize;
        let val_end = train_end + (n as f32 * val_ratio) as usize;

        let train = Self {
            x: self.x.slice(ndarray::s![..train_end, .., ..]).to_owned(),
            y: self.y[..train_end].to_vec(),
            timestamps: self.timestamps[..train_end].to_vec(),
            sequence_length: self.sequence_length,
            num_features: self.num_features,
            forward_window: self.forward_window,
            threshold: self.threshold,
        };

        let val = Self {
            x: self.x.slice(ndarray::s![train_end..val_end, .., ..]).to_owned(),
            y: self.y[train_end..val_end].to_vec(),
            timestamps: self.timestamps[train_end..val_end].to_vec(),
            sequence_length: self.sequence_length,
            num_features: self.num_features,
            forward_window: self.forward_window,
            threshold: self.threshold,
        };

        let test = Self {
            x: self.x.slice(ndarray::s![val_end.., .., ..]).to_owned(),
            y: self.y[val_end..].to_vec(),
            timestamps: self.timestamps[val_end..].to_vec(),
            sequence_length: self.sequence_length,
            num_features: self.num_features,
            forward_window: self.forward_window,
            threshold: self.threshold,
        };

        (train, val, test)
    }

    /// Create an iterator over batches
    pub fn batch_iter(&self, batch_size: usize, shuffle: bool) -> BatchIterator {
        let mut indices: Vec<usize> = (0..self.len()).collect();

        if shuffle {
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }

        BatchIterator {
            dataset: self,
            indices,
            batch_size,
            current_idx: 0,
        }
    }

    /// Get class distribution
    pub fn class_distribution(&self) -> [usize; 3] {
        let mut counts = [0usize; 3];
        for &label in &self.y {
            if (label as usize) < 3 {
                counts[label as usize] += 1;
            }
        }
        counts
    }

    /// Calculate class weights for imbalanced data
    pub fn class_weights(&self) -> [f32; 3] {
        let counts = self.class_distribution();
        let total = counts.iter().sum::<usize>() as f32;
        let n_classes = 3.0;

        [
            total / (n_classes * counts[0].max(1) as f32),
            total / (n_classes * counts[1].max(1) as f32),
            total / (n_classes * counts[2].max(1) as f32),
        ]
    }
}

/// Iterator over batches
pub struct BatchIterator<'a> {
    dataset: &'a Dataset,
    indices: Vec<usize>,
    batch_size: usize,
    current_idx: usize,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = (Array3<f32>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current_idx..end_idx];
        self.current_idx = end_idx;

        Some(self.dataset.get_batch(batch_indices))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| {
                let base = 50000.0 + (i as f64) * 10.0 * ((i % 5) as f64 - 2.0);
                Candle::new(
                    (i * 60000) as i64,
                    base,
                    base + 50.0,
                    base - 30.0,
                    base + 20.0,
                    1000.0,
                    50000000.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_dataset_creation() {
        let candles = create_test_candles(500);
        let dataset = Dataset::from_candles(candles, 256, 12, 0.002).unwrap();

        assert!(dataset.len() > 0);
        assert_eq!(dataset.num_features, 15);
        assert_eq!(dataset.sequence_length, 256);
    }

    #[test]
    fn test_dataset_split() {
        let candles = create_test_candles(500);
        let dataset = Dataset::from_candles(candles, 100, 10, 0.002).unwrap();
        let (train, val, test) = dataset.split(0.7, 0.15);

        assert!(train.len() > val.len());
        assert!(train.len() > test.len());
        assert_eq!(train.len() + val.len() + test.len(), dataset.len());
    }

    #[test]
    fn test_batch_iterator() {
        let candles = create_test_candles(500);
        let dataset = Dataset::from_candles(candles, 100, 10, 0.002).unwrap();

        let mut count = 0;
        for (x, y) in dataset.batch_iter(32, false) {
            count += y.len();
            assert_eq!(x.shape()[1], 15);
            assert_eq!(x.shape()[2], 100);
        }

        assert_eq!(count, dataset.len());
    }

    #[test]
    fn test_class_distribution() {
        let candles = create_test_candles(500);
        let dataset = Dataset::from_candles(candles, 100, 10, 0.002).unwrap();
        let dist = dataset.class_distribution();

        assert_eq!(dist.iter().sum::<usize>(), dataset.len());
    }
}
