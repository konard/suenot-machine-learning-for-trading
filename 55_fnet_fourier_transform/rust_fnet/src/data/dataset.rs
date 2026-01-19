//! Dataset utilities for sequence creation and batching.

use ndarray::{s, Array2, Array3};

/// Trading dataset for FNet model.
#[derive(Debug, Clone)]
pub struct TradingDataset {
    /// Sequence length
    pub seq_len: usize,
    /// Prediction horizon
    pub horizon: usize,
    /// Feature sequences [num_samples, seq_len, n_features]
    pub features: Array3<f64>,
    /// Target values [num_samples]
    pub targets: Vec<f64>,
    /// Corresponding prices for backtesting
    pub prices: Vec<f64>,
    /// Timestamps
    pub timestamps: Vec<i64>,
}

impl TradingDataset {
    /// Create sequences from feature matrix.
    ///
    /// # Arguments
    /// * `features` - Feature matrix [n_timesteps, n_features]
    /// * `prices` - Price array
    /// * `timestamps` - Timestamp array
    /// * `seq_len` - Sequence length for input
    /// * `horizon` - Prediction horizon
    /// * `target_col` - Column index for target (default: 0 = log_return)
    pub fn from_features(
        features: &Array2<f64>,
        prices: &[f64],
        timestamps: &[i64],
        seq_len: usize,
        horizon: usize,
        target_col: usize,
    ) -> Self {
        let n_timesteps = features.nrows();
        let n_features = features.ncols();

        // Calculate number of valid sequences
        let n_samples = n_timesteps.saturating_sub(seq_len + horizon);

        if n_samples == 0 {
            return Self {
                seq_len,
                horizon,
                features: Array3::zeros((0, seq_len, n_features)),
                targets: Vec::new(),
                prices: Vec::new(),
                timestamps: Vec::new(),
            };
        }

        // Create sequences
        let mut feature_seqs = Array3::zeros((n_samples, seq_len, n_features));
        let mut targets = Vec::with_capacity(n_samples);
        let mut sample_prices = Vec::with_capacity(n_samples);
        let mut sample_timestamps = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            // Extract sequence
            let seq = features.slice(s![i..i + seq_len, ..]);
            feature_seqs.slice_mut(s![i, .., ..]).assign(&seq);

            // Get target (cumulative return over horizon)
            let target_start = i + seq_len;
            let target_end = (target_start + horizon).min(n_timesteps);
            let target: f64 = features
                .slice(s![target_start..target_end, target_col])
                .sum();
            targets.push(target);

            // Store price at prediction point
            sample_prices.push(prices[i + seq_len]);
            sample_timestamps.push(timestamps[i + seq_len]);
        }

        Self {
            seq_len,
            horizon,
            features: feature_seqs,
            targets,
            prices: sample_prices,
            timestamps: sample_timestamps,
        }
    }

    /// Get number of samples.
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// Split dataset into train/validation/test.
    pub fn split(&self, train_ratio: f64, val_ratio: f64) -> (Self, Self, Self) {
        let n = self.len();
        let train_end = (n as f64 * train_ratio) as usize;
        let val_end = (n as f64 * (train_ratio + val_ratio)) as usize;

        let train = Self {
            seq_len: self.seq_len,
            horizon: self.horizon,
            features: self.features.slice(s![..train_end, .., ..]).to_owned(),
            targets: self.targets[..train_end].to_vec(),
            prices: self.prices[..train_end].to_vec(),
            timestamps: self.timestamps[..train_end].to_vec(),
        };

        let val = Self {
            seq_len: self.seq_len,
            horizon: self.horizon,
            features: self
                .features
                .slice(s![train_end..val_end, .., ..])
                .to_owned(),
            targets: self.targets[train_end..val_end].to_vec(),
            prices: self.prices[train_end..val_end].to_vec(),
            timestamps: self.timestamps[train_end..val_end].to_vec(),
        };

        let test = Self {
            seq_len: self.seq_len,
            horizon: self.horizon,
            features: self.features.slice(s![val_end.., .., ..]).to_owned(),
            targets: self.targets[val_end..].to_vec(),
            prices: self.prices[val_end..].to_vec(),
            timestamps: self.timestamps[val_end..].to_vec(),
        };

        (train, val, test)
    }

    /// Get a batch of samples.
    pub fn get_batch(&self, start: usize, batch_size: usize) -> (Array3<f64>, Vec<f64>) {
        let end = (start + batch_size).min(self.len());

        let batch_features = self.features.slice(s![start..end, .., ..]).to_owned();
        let batch_targets = self.targets[start..end].to_vec();

        (batch_features, batch_targets)
    }

    /// Iterate over batches.
    pub fn batches(&self, batch_size: usize) -> BatchIterator {
        BatchIterator {
            dataset: self,
            batch_size,
            current: 0,
        }
    }
}

/// Iterator over batches.
pub struct BatchIterator<'a> {
    dataset: &'a TradingDataset,
    batch_size: usize,
    current: usize,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = (Array3<f64>, Vec<f64>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.dataset.len() {
            return None;
        }

        let (features, targets) = self.dataset.get_batch(self.current, self.batch_size);
        self.current += self.batch_size;

        Some((features, targets))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_create_dataset() {
        let n = 100;
        let n_features = 8;

        let features = Array2::from_shape_fn((n, n_features), |(i, j)| (i * j) as f64 * 0.01);
        let prices: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let timestamps: Vec<i64> = (0..n).map(|i| 1700000000 + i as i64 * 3600).collect();

        let dataset = TradingDataset::from_features(&features, &prices, &timestamps, 20, 5, 0);

        assert_eq!(dataset.len(), n - 20 - 5);
        assert_eq!(dataset.features.dim().1, 20);
        assert_eq!(dataset.features.dim().2, 8);
    }

    #[test]
    fn test_split_dataset() {
        let n = 100;
        let n_features = 8;

        let features = Array2::from_shape_fn((n, n_features), |(i, j)| (i * j) as f64 * 0.01);
        let prices: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let timestamps: Vec<i64> = (0..n).map(|i| 1700000000 + i as i64 * 3600).collect();

        let dataset = TradingDataset::from_features(&features, &prices, &timestamps, 20, 5, 0);
        let (train, val, test) = dataset.split(0.7, 0.15);

        assert!(train.len() + val.len() + test.len() == dataset.len());
        assert!(train.len() > val.len());
        assert!(train.len() > test.len());
    }

    #[test]
    fn test_batch_iterator() {
        let n = 100;
        let n_features = 8;

        let features = Array2::from_shape_fn((n, n_features), |(i, j)| (i * j) as f64 * 0.01);
        let prices: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let timestamps: Vec<i64> = (0..n).map(|i| 1700000000 + i as i64 * 3600).collect();

        let dataset = TradingDataset::from_features(&features, &prices, &timestamps, 20, 5, 0);

        let batch_size = 16;
        let mut total_samples = 0;

        for (batch_features, batch_targets) in dataset.batches(batch_size) {
            assert_eq!(batch_features.dim().0, batch_targets.len());
            total_samples += batch_targets.len();
        }

        assert_eq!(total_samples, dataset.len());
    }
}
