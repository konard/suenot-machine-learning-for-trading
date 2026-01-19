//! Sequence data structures for training Linformer.

use ndarray::{Array2, Array3, s};
use rand::seq::SliceRandom;

/// Single sequence sample with features and target.
#[derive(Debug, Clone)]
pub struct SequenceData {
    /// Feature matrix [seq_len, n_features]
    pub features: Array2<f64>,
    /// Target values [prediction_horizon]
    pub targets: Vec<f64>,
    /// Target direction (1 = up, 0 = down)
    pub direction: i32,
}

/// Dataset containing multiple sequences for training.
pub struct SequenceDataset {
    /// All sequences
    pub sequences: Vec<SequenceData>,
    /// Sequence length
    pub seq_len: usize,
    /// Number of features
    pub n_features: usize,
    /// Prediction horizon
    pub prediction_horizon: usize,
}

impl SequenceDataset {
    /// Create sequences from feature matrix and price data.
    ///
    /// # Arguments
    /// * `features` - Feature matrix [n_samples, n_features]
    /// * `prices` - Price array for target calculation
    /// * `seq_len` - Length of each sequence
    /// * `prediction_horizon` - Number of future steps to predict
    pub fn from_features(
        features: &Array2<f64>,
        prices: &[f64],
        seq_len: usize,
        prediction_horizon: usize,
    ) -> Self {
        let (n_samples, n_features) = features.dim();
        let mut sequences = Vec::new();

        // Create sliding window sequences
        let end_idx = n_samples.saturating_sub(prediction_horizon);

        for i in seq_len..end_idx {
            // Extract feature sequence
            let feat_seq = features.slice(s![i - seq_len..i, ..]).to_owned();

            // Calculate target returns
            let current_price = prices[i - 1];
            let mut targets = Vec::with_capacity(prediction_horizon);

            for h in 1..=prediction_horizon {
                if i + h - 1 < prices.len() && current_price > 0.0 {
                    let future_price = prices[i + h - 1];
                    let return_pct = (future_price - current_price) / current_price;
                    targets.push(return_pct);
                } else {
                    targets.push(0.0);
                }
            }

            // Direction is based on first horizon step
            let direction = if !targets.is_empty() && targets[0] > 0.0 { 1 } else { 0 };

            sequences.push(SequenceData {
                features: feat_seq,
                targets,
                direction,
            });
        }

        Self {
            sequences,
            seq_len,
            n_features,
            prediction_horizon,
        }
    }

    /// Get the number of sequences in the dataset.
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Check if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Get a batch of sequences.
    pub fn get_batch(&self, indices: &[usize]) -> (Array3<f64>, Vec<Vec<f64>>, Vec<i32>) {
        let batch_size = indices.len();
        let mut features = Array3::zeros((batch_size, self.seq_len, self.n_features));
        let mut targets = Vec::with_capacity(batch_size);
        let mut directions = Vec::with_capacity(batch_size);

        for (batch_idx, &seq_idx) in indices.iter().enumerate() {
            let seq = &self.sequences[seq_idx];
            for i in 0..self.seq_len {
                for j in 0..self.n_features {
                    features[[batch_idx, i, j]] = seq.features[[i, j]];
                }
            }
            targets.push(seq.targets.clone());
            directions.push(seq.direction);
        }

        (features, targets, directions)
    }

    /// Split dataset into train and validation sets.
    pub fn train_val_split(&self, train_ratio: f64) -> (Vec<usize>, Vec<usize>) {
        let n = self.sequences.len();
        let train_size = (n as f64 * train_ratio) as usize;

        let train_indices: Vec<usize> = (0..train_size).collect();
        let val_indices: Vec<usize> = (train_size..n).collect();

        (train_indices, val_indices)
    }

    /// Shuffle indices for training.
    pub fn shuffle_indices(&self, indices: &mut [usize]) {
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
    }

    /// Get all features as a 3D array [n_sequences, seq_len, n_features].
    pub fn to_array(&self) -> Array3<f64> {
        let n = self.sequences.len();
        let mut arr = Array3::zeros((n, self.seq_len, self.n_features));

        for (idx, seq) in self.sequences.iter().enumerate() {
            for i in 0..self.seq_len {
                for j in 0..self.n_features {
                    arr[[idx, i, j]] = seq.features[[i, j]];
                }
            }
        }

        arr
    }

    /// Get all directions as a vector.
    pub fn get_directions(&self) -> Vec<i32> {
        self.sequences.iter().map(|s| s.direction).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_sequence_dataset_creation() {
        // Create sample data
        let n_samples = 100;
        let n_features = 5;
        let features = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
            (i * n_features + j) as f64
        });
        let prices: Vec<f64> = (0..n_samples).map(|i| 100.0 + i as f64).collect();

        let dataset = SequenceDataset::from_features(&features, &prices, 20, 5);

        assert!(!dataset.is_empty());
        assert_eq!(dataset.seq_len, 20);
        assert_eq!(dataset.n_features, 5);
    }

    #[test]
    fn test_batch_extraction() {
        let n_samples = 50;
        let n_features = 3;
        let features = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
            (i * n_features + j) as f64
        });
        let prices: Vec<f64> = (0..n_samples).map(|i| 100.0 + i as f64 * 0.1).collect();

        let dataset = SequenceDataset::from_features(&features, &prices, 10, 3);

        if dataset.len() >= 4 {
            let indices = vec![0, 1, 2, 3];
            let (batch_features, batch_targets, batch_directions) = dataset.get_batch(&indices);

            assert_eq!(batch_features.dim(), (4, 10, 3));
            assert_eq!(batch_targets.len(), 4);
            assert_eq!(batch_directions.len(), 4);
        }
    }

    #[test]
    fn test_train_val_split() {
        let n_samples = 100;
        let n_features = 2;
        let features = Array2::zeros((n_samples, n_features));
        let prices: Vec<f64> = vec![100.0; n_samples];

        let dataset = SequenceDataset::from_features(&features, &prices, 10, 1);
        let (train_idx, val_idx) = dataset.train_val_split(0.8);

        assert!(train_idx.len() + val_idx.len() == dataset.len());
    }
}
