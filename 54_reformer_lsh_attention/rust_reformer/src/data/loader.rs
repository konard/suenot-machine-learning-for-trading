//! Data loader for preparing datasets from kline data
//!
//! Handles conversion from raw market data to model-ready datasets.

use crate::api::{BybitError, Kline};
use crate::data::{Dataset, Sample};
use crate::data::features::{compute_all_features, log_returns};
use ndarray::Array2;

/// Data loader for preparing training and inference datasets
pub struct DataLoader {
    /// Whether to compute additional features
    compute_features: bool,
}

impl DataLoader {
    /// Create a new data loader
    pub fn new() -> Self {
        Self {
            compute_features: true,
        }
    }

    /// Create a data loader without feature computation
    pub fn raw() -> Self {
        Self {
            compute_features: false,
        }
    }

    /// Prepare a dataset from kline data
    ///
    /// # Arguments
    ///
    /// * `klines` - Historical kline data
    /// * `seq_len` - Input sequence length
    /// * `horizon` - Prediction horizon (future steps to predict)
    pub fn prepare_dataset(
        &self,
        klines: &[Kline],
        seq_len: usize,
        horizon: usize,
    ) -> Result<Dataset, BybitError> {
        let n = klines.len();

        if n < seq_len + horizon {
            return Err(BybitError::InsufficientData {
                need: seq_len + horizon,
                got: n,
            });
        }

        // Compute features
        let features = if self.compute_features {
            compute_all_features(klines)
        } else {
            // Use raw OHLCV
            self.raw_features(klines)
        };

        let feature_names = features.names.clone();
        let n_features = features.n_features();

        // Compute target: future log returns
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let returns = log_returns(&closes);

        // Create samples
        let mut dataset = Dataset::new(feature_names, seq_len, horizon);

        for i in 0..(n - seq_len - horizon + 1) {
            // Extract feature window
            let mut sample_features = Array2::zeros((seq_len, n_features));
            for t in 0..seq_len {
                for f in 0..n_features {
                    sample_features[[t, f]] = features.data[[i + t, f]];
                }
            }

            // Extract target (cumulative return over horizon)
            let target_start = i + seq_len;
            let mut target = Vec::with_capacity(horizon);

            for h in 0..horizon {
                let target_idx = target_start + h;
                if target_idx < returns.len() {
                    // Cumulative return from seq_len to seq_len + h
                    let cum_return: f64 = returns[target_start..=target_idx].iter().sum();
                    target.push(cum_return);
                } else {
                    target.push(0.0);
                }
            }

            let sample = Sample {
                features: sample_features,
                target,
                timestamp: klines[i + seq_len - 1].timestamp,
            };

            dataset.push(sample);
        }

        Ok(dataset)
    }

    /// Prepare raw OHLCV features without technical indicators
    fn raw_features(&self, klines: &[Kline]) -> crate::data::features::Features {
        let n = klines.len();
        let n_features = 5; // OHLCV

        let mut data = Array2::zeros((n, n_features));

        for (i, k) in klines.iter().enumerate() {
            data[[i, 0]] = k.open;
            data[[i, 1]] = k.high;
            data[[i, 2]] = k.low;
            data[[i, 3]] = k.close;
            data[[i, 4]] = k.volume;
        }

        // Normalize
        let data = self.normalize(data);

        crate::data::features::Features {
            data,
            names: vec![
                "open".to_string(),
                "high".to_string(),
                "low".to_string(),
                "close".to_string(),
                "volume".to_string(),
            ],
        }
    }

    /// Normalize feature matrix
    fn normalize(&self, mut data: Array2<f64>) -> Array2<f64> {
        let (n_rows, n_cols) = data.dim();

        for col in 0..n_cols {
            let column = data.column(col);
            let mean = column.mean().unwrap_or(0.0);
            let std = {
                let variance = column
                    .iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>()
                    / (n_rows - 1).max(1) as f64;
                variance.sqrt()
            };

            if std > 1e-10 {
                for row in 0..n_rows {
                    data[[row, col]] = (data[[row, col]] - mean) / std;
                }
            }
        }

        data
    }

    /// Prepare dataset for inference (single forward pass)
    pub fn prepare_inference(
        &self,
        klines: &[Kline],
        seq_len: usize,
    ) -> Result<Array2<f64>, BybitError> {
        if klines.len() < seq_len {
            return Err(BybitError::InsufficientData {
                need: seq_len,
                got: klines.len(),
            });
        }

        // Use last seq_len klines
        let recent_klines = &klines[klines.len() - seq_len..];

        let features = if self.compute_features {
            // Need more context for feature computation
            let context_start = klines.len().saturating_sub(seq_len + 50);
            let context_klines = &klines[context_start..];
            let all_features = compute_all_features(context_klines);

            // Extract last seq_len rows
            let start_row = all_features.data.nrows() - seq_len;
            let mut result = Array2::zeros((seq_len, all_features.n_features()));

            for t in 0..seq_len {
                for f in 0..all_features.n_features() {
                    result[[t, f]] = all_features.data[[start_row + t, f]];
                }
            }

            result
        } else {
            self.raw_features(recent_klines).data
        };

        Ok(features)
    }
}

impl Default for DataLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| Kline {
                timestamp: 1704067200000 + (i as u64 * 3600000),
                open: 42000.0 + (i as f64 * 10.0),
                high: 42100.0 + (i as f64 * 10.0),
                low: 41900.0 + (i as f64 * 10.0),
                close: 42050.0 + (i as f64 * 10.0),
                volume: 1000.0 + (i as f64 * 5.0),
                turnover: 42000000.0,
            })
            .collect()
    }

    #[test]
    fn test_prepare_dataset() {
        let klines = create_test_klines(200);
        let loader = DataLoader::new();

        let dataset = loader.prepare_dataset(&klines, 50, 10).unwrap();

        assert!(!dataset.is_empty());
        assert_eq!(dataset.seq_len, 50);
        assert_eq!(dataset.horizon, 10);

        // Each sample should have correct dimensions
        let sample = &dataset.samples[0];
        assert_eq!(sample.features.nrows(), 50);
        assert_eq!(sample.target.len(), 10);
    }

    #[test]
    fn test_insufficient_data() {
        let klines = create_test_klines(50);
        let loader = DataLoader::new();

        let result = loader.prepare_dataset(&klines, 100, 10);

        assert!(result.is_err());
    }

    #[test]
    fn test_prepare_inference() {
        let klines = create_test_klines(200);
        let loader = DataLoader::new();

        let features = loader.prepare_inference(&klines, 50).unwrap();

        assert_eq!(features.nrows(), 50);
    }

    #[test]
    fn test_raw_loader() {
        let klines = create_test_klines(100);
        let loader = DataLoader::raw();

        let dataset = loader.prepare_dataset(&klines, 20, 5).unwrap();

        // Raw features should be OHLCV (5 features)
        assert_eq!(dataset.n_features(), 5);
    }
}
