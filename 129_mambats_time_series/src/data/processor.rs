//! Data processing and dataset creation for time series forecasting.

use ndarray::{Array2, Array3, s};
use std::path::Path;

/// Time series dataset for training/inference
pub struct TimeSeriesDataset {
    /// Feature data (n_samples, n_features)
    pub features: Array2<f64>,
    /// Lookback window size
    pub lookback: usize,
    /// Forecast horizon
    pub forecast_horizon: usize,
    /// Stride for creating samples
    pub stride: usize,
}

impl TimeSeriesDataset {
    /// Create a new dataset
    pub fn new(
        features: Array2<f64>,
        lookback: usize,
        forecast_horizon: usize,
        stride: usize,
    ) -> Self {
        Self {
            features,
            lookback,
            forecast_horizon,
            stride,
        }
    }

    /// Get the number of valid samples
    pub fn len(&self) -> usize {
        let n_samples = self.features.nrows();
        if n_samples < self.lookback + self.forecast_horizon {
            return 0;
        }
        (n_samples - self.lookback - self.forecast_horizon) / self.stride + 1
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a single sample (input, target)
    pub fn get(&self, idx: usize) -> Option<(Array2<f64>, Array2<f64>)> {
        let start = idx * self.stride + self.lookback;
        let n_samples = self.features.nrows();

        if start + self.forecast_horizon > n_samples {
            return None;
        }

        let x = self.features.slice(s![start - self.lookback..start, ..]).to_owned();
        let y = self.features.slice(s![start..start + self.forecast_horizon, ..]).to_owned();

        Some((x, y))
    }

    /// Get all samples as batches
    pub fn to_batches(&self, batch_size: usize) -> Vec<(Array3<f64>, Array3<f64>)> {
        let n_samples = self.len();
        let n_features = self.features.ncols();
        let mut batches = Vec::new();

        for batch_start in (0..n_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n_samples);
            let actual_batch_size = batch_end - batch_start;

            let mut batch_x = Array3::zeros((actual_batch_size, self.lookback, n_features));
            let mut batch_y = Array3::zeros((actual_batch_size, self.forecast_horizon, n_features));

            for (i, idx) in (batch_start..batch_end).enumerate() {
                if let Some((x, y)) = self.get(idx) {
                    batch_x.slice_mut(s![i, .., ..]).assign(&x);
                    batch_y.slice_mut(s![i, .., ..]).assign(&y);
                }
            }

            batches.push((batch_x, batch_y));
        }

        batches
    }

    /// Split dataset into train, validation, and test sets
    pub fn train_val_test_split(
        &self,
        train_ratio: f64,
        val_ratio: f64,
    ) -> (TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset) {
        let n_samples = self.features.nrows();
        let train_end = (n_samples as f64 * train_ratio) as usize;
        let val_end = (n_samples as f64 * (train_ratio + val_ratio)) as usize;

        let train_features = self.features.slice(s![..train_end, ..]).to_owned();
        let val_features = self.features.slice(s![train_end..val_end, ..]).to_owned();
        let test_features = self.features.slice(s![val_end.., ..]).to_owned();

        (
            TimeSeriesDataset::new(train_features, self.lookback, self.forecast_horizon, self.stride),
            TimeSeriesDataset::new(val_features, self.lookback, self.forecast_horizon, self.stride),
            TimeSeriesDataset::new(test_features, self.lookback, self.forecast_horizon, self.stride),
        )
    }
}

/// Data processor for loading and preprocessing financial data
pub struct DataProcessor {
    /// Normalization statistics (mean, std) for each feature
    pub normalization_stats: Vec<(f64, f64)>,
}

impl DataProcessor {
    /// Create a new data processor
    pub fn new() -> Self {
        Self {
            normalization_stats: Vec::new(),
        }
    }

    /// Fit normalization statistics on data
    pub fn fit(&mut self, features: &Array2<f64>) {
        let n_features = features.ncols();
        self.normalization_stats.clear();

        for j in 0..n_features {
            let column: Vec<f64> = features
                .column(j)
                .iter()
                .filter(|&&x| !x.is_nan())
                .copied()
                .collect();

            if column.is_empty() {
                self.normalization_stats.push((0.0, 1.0));
                continue;
            }

            let mean = column.iter().sum::<f64>() / column.len() as f64;
            let variance: f64 =
                column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / column.len() as f64;
            let std = variance.sqrt().max(1e-8);

            self.normalization_stats.push((mean, std));
        }
    }

    /// Transform features using fitted statistics
    pub fn transform(&self, features: &Array2<f64>) -> Array2<f64> {
        let (n_samples, n_features) = features.dim();
        let mut normalized = Array2::zeros((n_samples, n_features));

        for j in 0..n_features {
            let (mean, std) = self.normalization_stats.get(j).copied().unwrap_or((0.0, 1.0));

            for i in 0..n_samples {
                let value = features[[i, j]];
                normalized[[i, j]] = if value.is_nan() {
                    0.0
                } else {
                    (value - mean) / std
                };
            }
        }

        normalized
    }

    /// Fit and transform
    pub fn fit_transform(&mut self, features: &Array2<f64>) -> Array2<f64> {
        self.fit(features);
        self.transform(features)
    }

    /// Inverse transform to original scale
    pub fn inverse_transform(&self, normalized: &Array2<f64>) -> Array2<f64> {
        let (n_samples, n_features) = normalized.dim();
        let mut original = Array2::zeros((n_samples, n_features));

        for j in 0..n_features {
            let (mean, std) = self.normalization_stats.get(j).copied().unwrap_or((0.0, 1.0));

            for i in 0..n_samples {
                original[[i, j]] = normalized[[i, j]] * std + mean;
            }
        }

        original
    }

    /// Save processor state to file
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let json = serde_json::to_string(&self.normalization_stats)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load processor state from file
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let normalization_stats: Vec<(f64, f64)> = serde_json::from_str(&json)?;
        Ok(Self { normalization_stats })
    }
}

impl Default for DataProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Remove rows with NaN values
pub fn remove_nan_rows(features: &Array2<f64>) -> Array2<f64> {
    let n_cols = features.ncols();
    let valid_rows: Vec<_> = features
        .rows()
        .into_iter()
        .filter(|row| row.iter().all(|&x| !x.is_nan()))
        .collect();

    let n_valid = valid_rows.len();
    if n_valid == 0 {
        return Array2::zeros((0, n_cols));
    }

    let mut result = Array2::zeros((n_valid, n_cols));
    for (i, row) in valid_rows.into_iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_dataset_len() {
        let features = Array2::zeros((100, 5));
        let dataset = TimeSeriesDataset::new(features, 10, 5, 1);
        assert_eq!(dataset.len(), 86);
    }

    #[test]
    fn test_dataset_get() {
        let features = Array2::from_shape_fn((20, 3), |(i, j)| (i * 3 + j) as f64);
        let dataset = TimeSeriesDataset::new(features, 5, 2, 1);

        let (x, y) = dataset.get(0).unwrap();
        assert_eq!(x.dim(), (5, 3));
        assert_eq!(y.dim(), (2, 3));
    }

    #[test]
    fn test_processor_fit_transform() {
        let features = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]];

        let mut processor = DataProcessor::new();
        let normalized = processor.fit_transform(&features);

        // Check that mean is approximately 0 and std is approximately 1
        for j in 0..2 {
            let col: Vec<f64> = normalized.column(j).iter().copied().collect();
            let mean: f64 = col.iter().sum::<f64>() / col.len() as f64;
            let std: f64 = (col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / col.len() as f64).sqrt();

            assert!((mean).abs() < 0.001);
            assert!((std - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_processor_inverse_transform() {
        let features = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];

        let mut processor = DataProcessor::new();
        let normalized = processor.fit_transform(&features);
        let recovered = processor.inverse_transform(&normalized);

        for i in 0..3 {
            for j in 0..2 {
                assert!((features[[i, j]] - recovered[[i, j]]).abs() < 0.001);
            }
        }
    }

    #[test]
    fn test_remove_nan_rows() {
        let features = array![
            [1.0, 2.0],
            [f64::NAN, 3.0],
            [4.0, 5.0],
            [6.0, f64::NAN],
            [7.0, 8.0]
        ];

        let cleaned = remove_nan_rows(&features);
        assert_eq!(cleaned.nrows(), 3);
    }
}
