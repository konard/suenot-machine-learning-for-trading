//! Data processing utilities for preparing ML datasets
//!
//! This module provides tools for cleaning, normalizing, and splitting
//! cryptocurrency market data for machine learning models.

use crate::api::bybit::Kline;
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

/// Processed dataset ready for ML models
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Feature matrix (n_samples x n_features)
    pub x: Array2<f64>,
    /// Target vector (n_samples)
    pub y: Array1<f64>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Timestamps for each sample
    pub timestamps: Vec<i64>,
}

impl Dataset {
    /// Create a new dataset
    pub fn new(
        x: Array2<f64>,
        y: Array1<f64>,
        feature_names: Vec<String>,
        timestamps: Vec<i64>,
    ) -> Self {
        Self {
            x,
            y,
            feature_names,
            timestamps,
        }
    }

    /// Get number of samples
    pub fn n_samples(&self) -> usize {
        self.x.nrows()
    }

    /// Get number of features
    pub fn n_features(&self) -> usize {
        self.x.ncols()
    }

    /// Split dataset into train and test sets
    pub fn train_test_split(&self, test_ratio: f64) -> (Dataset, Dataset) {
        let n = self.n_samples();
        let split_idx = ((1.0 - test_ratio) * n as f64) as usize;

        let x_train = self.x.slice(ndarray::s![..split_idx, ..]).to_owned();
        let x_test = self.x.slice(ndarray::s![split_idx.., ..]).to_owned();
        let y_train = self.y.slice(ndarray::s![..split_idx]).to_owned();
        let y_test = self.y.slice(ndarray::s![split_idx..]).to_owned();
        let timestamps_train = self.timestamps[..split_idx].to_vec();
        let timestamps_test = self.timestamps[split_idx..].to_vec();

        let train = Dataset::new(
            x_train,
            y_train,
            self.feature_names.clone(),
            timestamps_train,
        );
        let test = Dataset::new(x_test, y_test, self.feature_names.clone(), timestamps_test);

        (train, test)
    }
}

/// Data processor for preparing raw kline data for ML
#[derive(Debug, Default)]
pub struct DataProcessor {
    /// Means for each feature (for normalization)
    means: Option<Array1<f64>>,
    /// Standard deviations for each feature
    stds: Option<Array1<f64>>,
    /// Min values for each feature
    mins: Option<Array1<f64>>,
    /// Max values for each feature
    maxs: Option<Array1<f64>>,
}

impl DataProcessor {
    /// Create a new data processor
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate returns from kline data
    pub fn calculate_returns(klines: &[Kline]) -> Vec<f64> {
        if klines.len() < 2 {
            return vec![];
        }

        klines
            .windows(2)
            .map(|w| (w[1].close / w[0].close) - 1.0)
            .collect()
    }

    /// Calculate log returns
    pub fn calculate_log_returns(klines: &[Kline]) -> Vec<f64> {
        if klines.len() < 2 {
            return vec![];
        }

        klines
            .windows(2)
            .map(|w| (w[1].close / w[0].close).ln())
            .collect()
    }

    /// Calculate future returns (for target variable)
    pub fn calculate_future_returns(klines: &[Kline], periods: usize) -> Vec<f64> {
        if klines.len() <= periods {
            return vec![];
        }

        (0..klines.len() - periods)
            .map(|i| (klines[i + periods].close / klines[i].close) - 1.0)
            .collect()
    }

    /// Fit StandardScaler on data
    pub fn fit_standard_scaler(&mut self, x: &Array2<f64>) {
        let means = x.mean_axis(Axis(0)).unwrap();
        let stds = x.std_axis(Axis(0), 0.0);

        self.means = Some(means);
        self.stds = Some(stds);
    }

    /// Transform data using fitted StandardScaler
    pub fn transform_standard(&self, x: &Array2<f64>) -> Array2<f64> {
        let means = self.means.as_ref().expect("Scaler not fitted");
        let stds = self.stds.as_ref().expect("Scaler not fitted");

        let mut result = x.clone();
        for mut col in result.columns_mut() {
            for (i, val) in col.iter_mut().enumerate() {
                let std = stds[i % stds.len()];
                let mean = means[i % means.len()];
                if std > 1e-10 {
                    *val = (*val - mean) / std;
                } else {
                    *val = 0.0;
                }
            }
        }

        // Proper column-wise standardization
        let mut result = Array2::zeros(x.raw_dim());
        for (j, mut col) in result.columns_mut().into_iter().enumerate() {
            let std = stds[j];
            let mean = means[j];
            for (i, val) in col.iter_mut().enumerate() {
                if std > 1e-10 {
                    *val = (x[[i, j]] - mean) / std;
                } else {
                    *val = 0.0;
                }
            }
        }

        result
    }

    /// Fit MinMaxScaler on data
    pub fn fit_minmax_scaler(&mut self, x: &Array2<f64>) {
        let mins = x
            .columns()
            .into_iter()
            .map(|col| col.iter().cloned().fold(f64::INFINITY, f64::min))
            .collect::<Vec<_>>();
        let maxs = x
            .columns()
            .into_iter()
            .map(|col| col.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
            .collect::<Vec<_>>();

        self.mins = Some(Array1::from_vec(mins));
        self.maxs = Some(Array1::from_vec(maxs));
    }

    /// Transform data using fitted MinMaxScaler
    pub fn transform_minmax(&self, x: &Array2<f64>) -> Array2<f64> {
        let mins = self.mins.as_ref().expect("Scaler not fitted");
        let maxs = self.maxs.as_ref().expect("Scaler not fitted");

        let mut result = Array2::zeros(x.raw_dim());
        for (j, mut col) in result.columns_mut().into_iter().enumerate() {
            let min_val = mins[j];
            let max_val = maxs[j];
            let range = max_val - min_val;
            for (i, val) in col.iter_mut().enumerate() {
                if range > 1e-10 {
                    *val = (x[[i, j]] - min_val) / range;
                } else {
                    *val = 0.0;
                }
            }
        }

        result
    }

    /// Remove rows with NaN values
    pub fn dropna(x: &Array2<f64>, y: &Array1<f64>) -> (Array2<f64>, Array1<f64>) {
        let valid_rows: Vec<usize> = (0..x.nrows())
            .filter(|&i| {
                !x.row(i).iter().any(|v| v.is_nan())
                    && !y[i].is_nan()
                    && !x.row(i).iter().any(|v| v.is_infinite())
                    && !y[i].is_infinite()
            })
            .collect();

        let x_clean = Array2::from_shape_vec(
            (valid_rows.len(), x.ncols()),
            valid_rows
                .iter()
                .flat_map(|&i| x.row(i).to_vec())
                .collect(),
        )
        .unwrap();

        let y_clean = Array1::from_vec(valid_rows.iter().map(|&i| y[i]).collect());

        (x_clean, y_clean)
    }

    /// Compute correlation matrix
    pub fn correlation_matrix(x: &Array2<f64>) -> Array2<f64> {
        let n = x.ncols();
        let mut corr = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                let col_i = x.column(i);
                let col_j = x.column(j);

                let mean_i = col_i.mean().unwrap();
                let mean_j = col_j.mean().unwrap();

                let std_i = col_i.std(0.0);
                let std_j = col_j.std(0.0);

                if std_i > 1e-10 && std_j > 1e-10 {
                    let cov: f64 = col_i
                        .iter()
                        .zip(col_j.iter())
                        .map(|(a, b)| (a - mean_i) * (b - mean_j))
                        .sum::<f64>()
                        / (x.nrows() as f64);

                    corr[[i, j]] = cov / (std_i * std_j);
                } else {
                    corr[[i, j]] = if i == j { 1.0 } else { 0.0 };
                }
            }
        }

        corr
    }

    /// Add lag features to dataset
    pub fn add_lags(data: &[f64], lags: &[usize]) -> HashMap<String, Vec<f64>> {
        let mut result = HashMap::new();

        for &lag in lags {
            let lagged: Vec<f64> = std::iter::repeat(f64::NAN)
                .take(lag)
                .chain(data[..data.len() - lag].iter().cloned())
                .collect();

            result.insert(format!("lag_{}", lag), lagged);
        }

        result
    }
}

/// Train-test split utility
pub fn train_test_split(
    x: &Array2<f64>,
    y: &Array1<f64>,
    test_size: f64,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    let n = x.nrows();
    let split_idx = ((1.0 - test_size) * n as f64) as usize;

    let x_train = x.slice(ndarray::s![..split_idx, ..]).to_owned();
    let x_test = x.slice(ndarray::s![split_idx.., ..]).to_owned();
    let y_train = y.slice(ndarray::s![..split_idx]).to_owned();
    let y_test = y.slice(ndarray::s![split_idx..]).to_owned();

    (x_train, x_test, y_train, y_test)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_test_split() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let y = Array1::from_vec((0..10).map(|x| x as f64).collect());

        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2);

        assert_eq!(x_train.nrows(), 8);
        assert_eq!(x_test.nrows(), 2);
        assert_eq!(y_train.len(), 8);
        assert_eq!(y_test.len(), 2);
    }

    #[test]
    fn test_correlation_matrix() {
        let x = Array2::from_shape_vec((5, 2), vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0, 5.0, 10.0]).unwrap();
        let corr = DataProcessor::correlation_matrix(&x);

        // Perfect correlation
        assert!((corr[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((corr[[1, 0]] - 1.0).abs() < 1e-10);
    }
}
