//! Dataset for training and inference

use ndarray::Array3;
use crate::data::Features;

/// Dataset containing windowed sequences for the model
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Input sequences [batch, seq_len, features]
    pub inputs: Array3<f64>,
    /// Target values [batch]
    pub targets: Vec<f64>,
    /// Timestamps for each sequence
    pub timestamps: Vec<u64>,
    /// Close prices for backtesting
    pub close_prices: Vec<f64>,
    /// Sequence length
    pub seq_len: usize,
    /// Number of features
    pub num_features: usize,
}

impl Dataset {
    /// Create dataset from features with sliding window
    pub fn from_features(features: &Features, seq_len: usize) -> Self {
        let n = features.len();
        if n < seq_len {
            return Self::empty(seq_len, features.values.ncols());
        }

        let num_samples = n - seq_len + 1;
        let num_features = features.values.ncols();

        let mut inputs = Array3::zeros((num_samples, seq_len, num_features));
        let mut targets = Vec::with_capacity(num_samples);
        let mut timestamps = Vec::with_capacity(num_samples);
        let mut close_prices = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            // Copy sequence
            for j in 0..seq_len {
                for k in 0..num_features {
                    inputs[[i, j, k]] = features.values[[i + j, k]];
                }
            }

            // Target is at the end of sequence
            let target_idx = i + seq_len - 1;
            if let Some(ref t) = features.targets {
                if target_idx < t.len() {
                    targets.push(t[target_idx]);
                } else {
                    targets.push(0.0);
                }
            } else {
                targets.push(0.0);
            }

            timestamps.push(features.timestamps[target_idx]);
            close_prices.push(features.close_prices[target_idx]);
        }

        Self {
            inputs,
            targets,
            timestamps,
            close_prices,
            seq_len,
            num_features,
        }
    }

    /// Create empty dataset
    pub fn empty(seq_len: usize, num_features: usize) -> Self {
        Self {
            inputs: Array3::zeros((0, seq_len, num_features)),
            targets: Vec::new(),
            timestamps: Vec::new(),
            close_prices: Vec::new(),
            seq_len,
            num_features,
        }
    }

    /// Number of samples in the dataset
    pub fn len(&self) -> usize {
        self.inputs.dim().0
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }

    /// Get a batch of samples
    pub fn get_batch(&self, indices: &[usize]) -> (Array3<f64>, Vec<f64>) {
        let batch_size = indices.len();
        let mut batch_inputs = Array3::zeros((batch_size, self.seq_len, self.num_features));
        let mut batch_targets = Vec::with_capacity(batch_size);

        for (b, &idx) in indices.iter().enumerate() {
            for j in 0..self.seq_len {
                for k in 0..self.num_features {
                    batch_inputs[[b, j, k]] = self.inputs[[idx, j, k]];
                }
            }
            batch_targets.push(self.targets[idx]);
        }

        (batch_inputs, batch_targets)
    }

    /// Generate random batch indices
    pub fn random_batch_indices(&self, batch_size: usize) -> Vec<usize> {
        use rand::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..self.len()).collect();
        indices.shuffle(&mut rand::thread_rng());
        indices.truncate(batch_size);
        indices
    }

    /// Split dataset into training batches
    pub fn iter_batches(&self, batch_size: usize) -> impl Iterator<Item = (Array3<f64>, Vec<f64>)> + '_ {
        (0..self.len())
            .step_by(batch_size)
            .map(move |start| {
                let end = (start + batch_size).min(self.len());
                let indices: Vec<usize> = (start..end).collect();
                self.get_batch(&indices)
            })
    }

    /// Train/test split
    pub fn train_test_split(&self, train_ratio: f64) -> (Dataset, Dataset) {
        let n = self.len();
        let train_size = ((n as f64) * train_ratio) as usize;

        let train = Dataset {
            inputs: self.inputs.slice(ndarray::s![..train_size, .., ..]).to_owned(),
            targets: self.targets[..train_size].to_vec(),
            timestamps: self.timestamps[..train_size].to_vec(),
            close_prices: self.close_prices[..train_size].to_vec(),
            seq_len: self.seq_len,
            num_features: self.num_features,
        };

        let test = Dataset {
            inputs: self.inputs.slice(ndarray::s![train_size.., .., ..]).to_owned(),
            targets: self.targets[train_size..].to_vec(),
            timestamps: self.timestamps[train_size..].to_vec(),
            close_prices: self.close_prices[train_size..].to_vec(),
            seq_len: self.seq_len,
            num_features: self.num_features,
        };

        (train, test)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Kline;

    fn create_test_features(n: usize) -> Features {
        let klines: Vec<Kline> = (0..n)
            .map(|i| Kline {
                timestamp: i as u64 * 3600000,
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0,
                turnover: 100000.0,
            })
            .collect();

        Features::from_klines(&klines, 24)
    }

    #[test]
    fn test_dataset_from_features() {
        let features = create_test_features(200);
        let dataset = Dataset::from_features(&features, 64);

        assert!(dataset.len() > 0);
        assert_eq!(dataset.inputs.dim().1, 64);
        assert_eq!(dataset.inputs.dim().2, features.values.ncols());
    }

    #[test]
    fn test_get_batch() {
        let features = create_test_features(200);
        let dataset = Dataset::from_features(&features, 32);

        let (batch_inputs, batch_targets) = dataset.get_batch(&[0, 1, 2]);

        assert_eq!(batch_inputs.dim().0, 3);
        assert_eq!(batch_targets.len(), 3);
    }

    #[test]
    fn test_train_test_split() {
        let features = create_test_features(200);
        let dataset = Dataset::from_features(&features, 32);

        let (train, test) = dataset.train_test_split(0.8);

        assert_eq!(train.len() + test.len(), dataset.len());
        assert!(train.len() > test.len());
    }

    #[test]
    fn test_iter_batches() {
        let features = create_test_features(200);
        let dataset = Dataset::from_features(&features, 32);

        let batch_count = dataset.iter_batches(16).count();
        assert!(batch_count > 0);
    }
}
