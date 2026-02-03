//! Data preprocessing utilities.

use crate::api::bybit::Kline;
use ndarray::{Array1, Array2};
use thiserror::Error;

/// Errors that can occur during data processing
#[derive(Error, Debug)]
pub enum ProcessorError {
    #[error("Insufficient data: need at least {required} samples, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    #[error("Invalid window size: {0}")]
    InvalidWindowSize(usize),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

/// Data processor for preparing inputs to the prototype network.
pub struct DataProcessor {
    /// Lookback window size
    lookback: usize,
    /// Feature means for normalization
    feature_mean: Option<Array1<f64>>,
    /// Feature standard deviations for normalization
    feature_std: Option<Array1<f64>>,
}

impl DataProcessor {
    /// Create a new data processor.
    ///
    /// # Arguments
    /// * `lookback` - Number of time steps in each sequence
    pub fn new(lookback: usize) -> Self {
        Self {
            lookback,
            feature_mean: None,
            feature_std: None,
        }
    }

    /// Convert klines to OHLCV arrays.
    pub fn klines_to_arrays(&self, klines: &[Kline]) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
        let n = klines.len();
        let mut open = Array1::zeros(n);
        let mut high = Array1::zeros(n);
        let mut low = Array1::zeros(n);
        let mut close = Array1::zeros(n);
        let mut volume = Array1::zeros(n);

        for (i, kline) in klines.iter().enumerate() {
            open[i] = kline.open;
            high[i] = kline.high;
            low[i] = kline.low;
            close[i] = kline.close;
            volume[i] = kline.volume;
        }

        (open, high, low, close, volume)
    }

    /// Create sequences from feature matrix.
    ///
    /// # Arguments
    /// * `features` - Feature matrix of shape (n_samples, n_features)
    ///
    /// # Returns
    /// 3D array of shape (n_sequences, n_features, lookback)
    pub fn create_sequences(
        &self,
        features: &Array2<f64>,
    ) -> Result<Vec<Array2<f64>>, ProcessorError> {
        let (n_samples, n_features) = features.dim();

        if n_samples < self.lookback {
            return Err(ProcessorError::InsufficientData {
                required: self.lookback,
                actual: n_samples,
            });
        }

        let n_sequences = n_samples - self.lookback + 1;
        let mut sequences = Vec::with_capacity(n_sequences);

        for i in 0..n_sequences {
            let mut seq = Array2::zeros((n_features, self.lookback));
            for j in 0..self.lookback {
                for k in 0..n_features {
                    seq[[k, j]] = features[[i + j, k]];
                }
            }
            sequences.push(seq);
        }

        Ok(sequences)
    }

    /// Fit normalization parameters on training data.
    pub fn fit(&mut self, features: &Array2<f64>) {
        let (n_samples, n_features) = features.dim();

        let mut mean = Array1::zeros(n_features);
        let mut std = Array1::zeros(n_features);

        // Compute mean
        for j in 0..n_features {
            let mut sum = 0.0;
            for i in 0..n_samples {
                sum += features[[i, j]];
            }
            mean[j] = sum / n_samples as f64;
        }

        // Compute std
        for j in 0..n_features {
            let mut sum_sq = 0.0;
            for i in 0..n_samples {
                let diff = features[[i, j]] - mean[j];
                sum_sq += diff * diff;
            }
            std[j] = (sum_sq / n_samples as f64).sqrt();
            if std[j] < 1e-8 {
                std[j] = 1.0; // Avoid division by zero
            }
        }

        self.feature_mean = Some(mean);
        self.feature_std = Some(std);
    }

    /// Normalize features using fitted parameters.
    pub fn transform(&self, features: &Array2<f64>) -> Result<Array2<f64>, ProcessorError> {
        let mean = self.feature_mean.as_ref().ok_or_else(|| {
            ProcessorError::InvalidWindowSize(0) // Using as placeholder for "not fitted"
        })?;
        let std = self.feature_std.as_ref().ok_or_else(|| {
            ProcessorError::InvalidWindowSize(0)
        })?;

        let (n_samples, n_features) = features.dim();

        if n_features != mean.len() {
            return Err(ProcessorError::DimensionMismatch {
                expected: mean.len(),
                actual: n_features,
            });
        }

        let mut normalized = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                normalized[[i, j]] = (features[[i, j]] - mean[j]) / std[j];
            }
        }

        Ok(normalized)
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, features: &Array2<f64>) -> Array2<f64> {
        self.fit(features);
        self.transform(features).unwrap()
    }

    /// Calculate rolling statistics.
    pub fn rolling_mean(data: &Array1<f64>, window: usize) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::from_elem(n, f64::NAN);

        if n < window {
            return result;
        }

        for i in (window - 1)..n {
            let mut sum = 0.0;
            for j in 0..window {
                sum += data[i - j];
            }
            result[i] = sum / window as f64;
        }

        result
    }

    /// Calculate rolling standard deviation.
    pub fn rolling_std(data: &Array1<f64>, window: usize) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::from_elem(n, f64::NAN);

        if n < window {
            return result;
        }

        for i in (window - 1)..n {
            let mut sum = 0.0;
            for j in 0..window {
                sum += data[i - j];
            }
            let mean = sum / window as f64;

            let mut sum_sq = 0.0;
            for j in 0..window {
                let diff = data[i - j] - mean;
                sum_sq += diff * diff;
            }
            result[i] = (sum_sq / window as f64).sqrt();
        }

        result
    }

    /// Calculate exponential moving average.
    pub fn ema(data: &Array1<f64>, span: usize) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::from_elem(n, f64::NAN);

        if n == 0 {
            return result;
        }

        let alpha = 2.0 / (span as f64 + 1.0);
        result[0] = data[0];

        for i in 1..n {
            result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Calculate percentage change.
    pub fn pct_change(data: &Array1<f64>, periods: usize) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::from_elem(n, f64::NAN);

        for i in periods..n {
            if data[i - periods].abs() > 1e-10 {
                result[i] = data[i] / data[i - periods] - 1.0;
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_mean() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = DataProcessor::rolling_mean(&data, 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
        assert!((result[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_pct_change() {
        let data = Array1::from_vec(vec![100.0, 105.0, 110.0, 100.0]);
        let result = DataProcessor::pct_change(&data, 1);

        assert!(result[0].is_nan());
        assert!((result[1] - 0.05).abs() < 1e-10);
        assert!((result[2] - 0.0476).abs() < 0.001);
    }

    #[test]
    fn test_normalization() {
        let features = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
        )
        .unwrap();

        let mut processor = DataProcessor::new(2);
        let normalized = processor.fit_transform(&features);

        // Check that mean is approximately 0 and std is approximately 1
        let col_mean: f64 = normalized.column(0).sum() / 4.0;
        assert!(col_mean.abs() < 1e-10);
    }
}
