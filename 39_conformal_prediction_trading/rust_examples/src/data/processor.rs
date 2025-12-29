//! Data preprocessing utilities
//!
//! Provides functions for data cleaning, normalization, and train/test splitting.

use ndarray::{Array1, Array2, Axis};

/// Data processor for preprocessing features and targets
pub struct DataProcessor;

impl DataProcessor {
    /// Split data into train, calibration, and test sets
    ///
    /// # Arguments
    /// * `features` - Feature matrix (n_samples x n_features)
    /// * `targets` - Target vector (n_samples)
    /// * `train_ratio` - Fraction of data for training (e.g., 0.6)
    /// * `calib_ratio` - Fraction of data for calibration (e.g., 0.2)
    ///
    /// Returns: ((X_train, y_train), (X_calib, y_calib), (X_test, y_test))
    pub fn train_calib_test_split(
        features: &Array2<f64>,
        targets: &[f64],
        train_ratio: f64,
        calib_ratio: f64,
    ) -> (
        (Array2<f64>, Vec<f64>),
        (Array2<f64>, Vec<f64>),
        (Array2<f64>, Vec<f64>),
    ) {
        let n = features.nrows();
        let train_end = (n as f64 * train_ratio) as usize;
        let calib_end = train_end + (n as f64 * calib_ratio) as usize;

        let x_train = features.slice(ndarray::s![..train_end, ..]).to_owned();
        let y_train = targets[..train_end].to_vec();

        let x_calib = features.slice(ndarray::s![train_end..calib_end, ..]).to_owned();
        let y_calib = targets[train_end..calib_end].to_vec();

        let x_test = features.slice(ndarray::s![calib_end.., ..]).to_owned();
        let y_test = targets[calib_end..].to_vec();

        ((x_train, y_train), (x_calib, y_calib), (x_test, y_test))
    }

    /// Split data for time series with a rolling window
    ///
    /// Returns indices for train, calibration, and test at each step
    pub fn rolling_split(
        n_samples: usize,
        train_size: usize,
        calib_size: usize,
    ) -> Vec<(std::ops::Range<usize>, std::ops::Range<usize>, usize)> {
        let mut splits = Vec::new();
        let start = train_size + calib_size;

        for test_idx in start..n_samples {
            let train_start = test_idx - train_size - calib_size;
            let calib_start = test_idx - calib_size;

            splits.push((
                train_start..calib_start,
                calib_start..test_idx,
                test_idx,
            ));
        }

        splits
    }

    /// Standardize features (zero mean, unit variance)
    ///
    /// Returns standardized features, means, and standard deviations
    pub fn standardize(features: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let n_features = features.ncols();
        let mut means = Array1::zeros(n_features);
        let mut stds = Array1::zeros(n_features);
        let mut standardized = features.clone();

        for j in 0..n_features {
            let col = features.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let std = Self::std_dev(&col.to_vec());

            means[j] = mean;
            stds[j] = if std.abs() < 1e-10 { 1.0 } else { std };

            for i in 0..features.nrows() {
                standardized[[i, j]] = (features[[i, j]] - mean) / stds[j];
            }
        }

        (standardized, means, stds)
    }

    /// Apply pre-computed standardization to new data
    pub fn apply_standardization(
        features: &Array2<f64>,
        means: &Array1<f64>,
        stds: &Array1<f64>,
    ) -> Array2<f64> {
        let mut standardized = features.clone();

        for j in 0..features.ncols() {
            for i in 0..features.nrows() {
                standardized[[i, j]] = (features[[i, j]] - means[j]) / stds[j];
            }
        }

        standardized
    }

    /// Calculate standard deviation
    pub fn std_dev(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }

    /// Remove rows with NaN or Inf values
    pub fn remove_invalid_rows(features: &Array2<f64>, targets: &[f64]) -> (Array2<f64>, Vec<f64>) {
        let mut valid_indices = Vec::new();

        for i in 0..features.nrows() {
            let row_valid = features.row(i).iter().all(|&x| x.is_finite());
            let target_valid = targets.get(i).map(|&x| x.is_finite()).unwrap_or(false);

            if row_valid && target_valid {
                valid_indices.push(i);
            }
        }

        let n_valid = valid_indices.len();
        let n_features = features.ncols();

        let mut clean_features = Array2::zeros((n_valid, n_features));
        let mut clean_targets = Vec::with_capacity(n_valid);

        for (new_idx, &old_idx) in valid_indices.iter().enumerate() {
            for j in 0..n_features {
                clean_features[[new_idx, j]] = features[[old_idx, j]];
            }
            clean_targets.push(targets[old_idx]);
        }

        (clean_features, clean_targets)
    }

    /// Clip extreme values (winsorization)
    pub fn clip_features(features: &Array2<f64>, lower_pct: f64, upper_pct: f64) -> Array2<f64> {
        let mut clipped = features.clone();

        for j in 0..features.ncols() {
            let col: Vec<f64> = features.column(j).iter().copied().collect();
            let (lower, upper) = Self::percentiles(&col, lower_pct, upper_pct);

            for i in 0..features.nrows() {
                clipped[[i, j]] = clipped[[i, j]].max(lower).min(upper);
            }
        }

        clipped
    }

    /// Calculate percentiles
    pub fn percentiles(values: &[f64], lower_pct: f64, upper_pct: f64) -> (f64, f64) {
        if values.is_empty() {
            return (0.0, 0.0);
        }

        let mut sorted: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if sorted.is_empty() {
            return (0.0, 0.0);
        }

        let lower_idx = ((sorted.len() as f64 - 1.0) * lower_pct / 100.0) as usize;
        let upper_idx = ((sorted.len() as f64 - 1.0) * upper_pct / 100.0) as usize;

        (sorted[lower_idx], sorted[upper_idx.min(sorted.len() - 1)])
    }

    /// Calculate quantile
    pub fn quantile(values: &[f64], q: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if sorted.is_empty() {
            return 0.0;
        }

        let idx = ((sorted.len() as f64 - 1.0) * q) as usize;
        let idx = idx.min(sorted.len() - 1);

        sorted[idx]
    }

    /// Lag features by n periods (for creating lagged features)
    pub fn lag_features(features: &Array2<f64>, lag: usize) -> Array2<f64> {
        let (n, m) = features.dim();
        if lag >= n {
            return Array2::zeros((0, m));
        }

        features.slice(ndarray::s![..(n - lag), ..]).to_owned()
    }

    /// Get targets aligned with lagged features
    pub fn align_targets(targets: &[f64], lag: usize) -> Vec<f64> {
        if lag >= targets.len() {
            return vec![];
        }
        targets[lag..].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_train_calib_test_split() {
        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let ((x_train, y_train), (x_calib, y_calib), (x_test, y_test)) =
            DataProcessor::train_calib_test_split(&features, &targets, 0.4, 0.4);

        assert_eq!(x_train.nrows(), 2);
        assert_eq!(x_calib.nrows(), 2);
        assert_eq!(x_test.nrows(), 1);
    }

    #[test]
    fn test_standardize() {
        let features = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let (standardized, means, stds) = DataProcessor::standardize(&features);

        // Check means are approximately zero
        for j in 0..standardized.ncols() {
            let col_mean: f64 = standardized.column(j).mean().unwrap();
            assert!(col_mean.abs() < 1e-10);
        }
    }

    #[test]
    fn test_quantile() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let q50 = DataProcessor::quantile(&values, 0.5);
        assert!((q50 - 5.0).abs() < 0.1 || (q50 - 6.0).abs() < 0.1);

        let q90 = DataProcessor::quantile(&values, 0.9);
        assert!((q90 - 9.0).abs() < 0.5);
    }

    #[test]
    fn test_rolling_split() {
        let splits = DataProcessor::rolling_split(10, 5, 2);

        // First split: train 0..5, calib 5..7, test 7
        assert_eq!(splits[0].0, 0..5);
        assert_eq!(splits[0].1, 5..7);
        assert_eq!(splits[0].2, 7);
    }
}
