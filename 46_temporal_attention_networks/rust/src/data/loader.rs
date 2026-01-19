//! Data loader for TABL model

use crate::api::{BybitClient, Kline};
use crate::data::features::{prepare_features, Features};
use ndarray::{Array2, Array3};

/// Data loader for preparing TABL training data
pub struct DataLoader {
    /// Sequence length for input windows
    seq_len: usize,
    /// Prediction horizon
    horizon: usize,
    /// Classification threshold for labeling
    threshold: f64,
}

impl Default for DataLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DataLoader {
    /// Create a new data loader with default parameters
    pub fn new() -> Self {
        Self {
            seq_len: 100,
            horizon: 10,
            threshold: 0.0002,
        }
    }

    /// Create a data loader with custom parameters
    pub fn with_params(seq_len: usize, horizon: usize, threshold: f64) -> Self {
        Self {
            seq_len,
            horizon,
            threshold,
        }
    }

    /// Set the sequence length
    pub fn seq_len(mut self, seq_len: usize) -> Self {
        self.seq_len = seq_len;
        self
    }

    /// Set the prediction horizon
    pub fn horizon(mut self, horizon: usize) -> Self {
        self.horizon = horizon;
        self
    }

    /// Set the classification threshold
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Prepare data for TABL from klines
    ///
    /// # Returns
    /// * `(X, y)` - Feature tensor [n_samples, seq_len, n_features] and labels
    pub fn prepare_tabl_data(&self, klines: &[Kline]) -> anyhow::Result<(Array3<f64>, Vec<i32>)> {
        if klines.len() < self.seq_len + self.horizon {
            anyhow::bail!(
                "Not enough data: need at least {} candles, got {}",
                self.seq_len + self.horizon,
                klines.len()
            );
        }

        // Compute features
        let mut features = prepare_features(klines);
        features.normalize();

        // Create labels based on future returns
        let close_prices: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let labels = self.create_labels(&close_prices);

        // Create sliding windows
        let n_samples = klines.len() - self.seq_len - self.horizon + 1;
        let n_features = features.n_features();

        let mut x = Array3::zeros((n_samples, self.seq_len, n_features));
        let mut y = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            // Copy feature window
            for t in 0..self.seq_len {
                for f in 0..n_features {
                    x[[i, t, f]] = features.data[[i + t, f]];
                }
            }
            // Get label (at the end of the sequence)
            y.push(labels[i + self.seq_len - 1]);
        }

        Ok((x, y))
    }

    /// Create classification labels based on future returns
    fn create_labels(&self, prices: &[f64]) -> Vec<i32> {
        let mut labels = vec![1; prices.len()]; // Default to "hold"

        for i in 0..prices.len() - self.horizon {
            let current_price = prices[i];
            let future_price = prices[i + self.horizon];
            let return_pct = (future_price - current_price) / current_price;

            labels[i] = if return_pct > self.threshold {
                2 // Up
            } else if return_pct < -self.threshold {
                0 // Down
            } else {
                1 // Hold
            };
        }

        labels
    }

    /// Load data from Bybit API and prepare for TABL
    pub async fn load_from_bybit(
        &self,
        client: &BybitClient,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<(Array3<f64>, Vec<i32>)> {
        let klines = client.get_klines(symbol, interval, limit).await?;
        self.prepare_tabl_data(&klines)
    }

    /// Split data into train/validation/test sets
    pub fn train_val_test_split(
        &self,
        x: &Array3<f64>,
        y: &[i32],
        train_ratio: f64,
        val_ratio: f64,
    ) -> (
        (Array3<f64>, Vec<i32>),
        (Array3<f64>, Vec<i32>),
        (Array3<f64>, Vec<i32>),
    ) {
        let n_samples = x.shape()[0];
        let train_end = (n_samples as f64 * train_ratio) as usize;
        let val_end = (n_samples as f64 * (train_ratio + val_ratio)) as usize;

        let x_train = x.slice(ndarray::s![..train_end, .., ..]).to_owned();
        let y_train = y[..train_end].to_vec();

        let x_val = x.slice(ndarray::s![train_end..val_end, .., ..]).to_owned();
        let y_val = y[train_end..val_end].to_vec();

        let x_test = x.slice(ndarray::s![val_end.., .., ..]).to_owned();
        let y_test = y[val_end..].to_vec();

        ((x_train, y_train), (x_val, y_val), (x_test, y_test))
    }

    /// Create batches for training
    pub fn create_batches(
        x: &Array3<f64>,
        y: &[i32],
        batch_size: usize,
    ) -> Vec<(Array3<f64>, Vec<i32>)> {
        let n_samples = x.shape()[0];
        let n_batches = (n_samples + batch_size - 1) / batch_size;

        (0..n_batches)
            .map(|i| {
                let start = i * batch_size;
                let end = (start + batch_size).min(n_samples);

                let x_batch = x.slice(ndarray::s![start..end, .., ..]).to_owned();
                let y_batch = y[start..end].to_vec();

                (x_batch, y_batch)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| {
                let price = 100.0 + (i as f64 * 0.05).sin() * 5.0;
                Kline {
                    start_time: i as i64 * 3600000,
                    open: price,
                    high: price + 1.0,
                    low: price - 1.0,
                    close: price + (i as f64 * 0.1).sin(),
                    volume: 1000.0 + (i as f64 * 10.0),
                    turnover: 100000.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_prepare_tabl_data() {
        let klines = create_test_klines(200);
        let loader = DataLoader::new();
        let result = loader.prepare_tabl_data(&klines);

        assert!(result.is_ok());
        let (x, y) = result.unwrap();

        // Check dimensions
        assert_eq!(x.shape()[1], 100); // seq_len
        assert_eq!(x.shape()[2], 6);   // n_features
        assert_eq!(x.shape()[0], y.len());
    }

    #[test]
    fn test_create_labels() {
        let prices = vec![100.0, 100.1, 99.9, 100.2, 100.3, 100.0, 99.8, 100.5, 100.4, 100.2];
        let loader = DataLoader::with_params(3, 2, 0.001);
        let labels = loader.create_labels(&prices);

        assert_eq!(labels.len(), prices.len());
        // Labels should be 0, 1, or 2
        assert!(labels.iter().all(|&l| l >= 0 && l <= 2));
    }

    #[test]
    fn test_train_val_test_split() {
        let klines = create_test_klines(300);
        let loader = DataLoader::with_params(50, 5, 0.0002);
        let (x, y) = loader.prepare_tabl_data(&klines).unwrap();

        let ((x_train, y_train), (x_val, y_val), (x_test, y_test)) =
            loader.train_val_test_split(&x, &y, 0.7, 0.15);

        let total = y_train.len() + y_val.len() + y_test.len();
        assert_eq!(total, y.len());

        // Roughly check proportions
        let train_ratio = y_train.len() as f64 / total as f64;
        assert!(train_ratio > 0.6 && train_ratio < 0.8);
    }

    #[test]
    fn test_create_batches() {
        let klines = create_test_klines(200);
        let loader = DataLoader::new();
        let (x, y) = loader.prepare_tabl_data(&klines).unwrap();

        let batches = DataLoader::create_batches(&x, &y, 16);

        // Check that all samples are covered
        let total_samples: usize = batches.iter().map(|(_, y)| y.len()).sum();
        assert_eq!(total_samples, y.len());
    }
}
