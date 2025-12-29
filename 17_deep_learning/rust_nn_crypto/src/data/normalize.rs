//! Data Normalization Utilities
//!
//! Provides normalization methods for neural network input

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Normalizer trait
pub trait Normalizer: Send + Sync {
    /// Fit the normalizer to training data
    fn fit(&mut self, data: &Array2<f64>);

    /// Transform data using fitted parameters
    fn transform(&self, data: &Array2<f64>) -> Array2<f64>;

    /// Fit and transform in one step
    fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
        self.fit(data);
        self.transform(data)
    }

    /// Inverse transform to original scale
    fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64>;
}

/// Min-Max normalization to [0, 1] range
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinMaxNormalizer {
    pub min: Option<Array1<f64>>,
    pub max: Option<Array1<f64>>,
    pub range: Option<Array1<f64>>,
    pub feature_range: (f64, f64),
}

impl Default for MinMaxNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl MinMaxNormalizer {
    pub fn new() -> Self {
        Self {
            min: None,
            max: None,
            range: None,
            feature_range: (0.0, 1.0),
        }
    }

    /// Set custom feature range (default is [0, 1])
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.feature_range = (min, max);
        self
    }
}

impl Normalizer for MinMaxNormalizer {
    fn fit(&mut self, data: &Array2<f64>) {
        let min = data.fold_axis(Axis(0), f64::INFINITY, |&a, &b| a.min(b));
        let max = data.fold_axis(Axis(0), f64::NEG_INFINITY, |&a, &b| a.max(b));
        let range = &max - &min;

        // Avoid division by zero
        let range = range.mapv(|v| if v.abs() < 1e-10 { 1.0 } else { v });

        self.min = Some(min);
        self.max = Some(max);
        self.range = Some(range);
    }

    fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let min = self.min.as_ref().expect("Normalizer not fitted");
        let range = self.range.as_ref().expect("Normalizer not fitted");
        let (out_min, out_max) = self.feature_range;
        let out_range = out_max - out_min;

        let mut result = Array2::zeros(data.dim());
        for (i, row) in data.rows().into_iter().enumerate() {
            let normalized = (&row - min) / range;
            let scaled = &normalized * out_range + out_min;
            result.row_mut(i).assign(&scaled);
        }
        result
    }

    fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let min = self.min.as_ref().expect("Normalizer not fitted");
        let range = self.range.as_ref().expect("Normalizer not fitted");
        let (out_min, out_max) = self.feature_range;
        let out_range = out_max - out_min;

        let mut result = Array2::zeros(data.dim());
        for (i, row) in data.rows().into_iter().enumerate() {
            let unscaled = (&row - out_min) / out_range;
            let denormalized = &unscaled * range + min;
            result.row_mut(i).assign(&denormalized);
        }
        result
    }
}

/// Standard (Z-score) normalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardNormalizer {
    pub mean: Option<Array1<f64>>,
    pub std: Option<Array1<f64>>,
}

impl Default for StandardNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl StandardNormalizer {
    pub fn new() -> Self {
        Self {
            mean: None,
            std: None,
        }
    }
}

impl Normalizer for StandardNormalizer {
    fn fit(&mut self, data: &Array2<f64>) {
        let n = data.nrows() as f64;

        // Calculate mean
        let mean = data.sum_axis(Axis(0)) / n;

        // Calculate standard deviation
        let mut std = Array1::zeros(data.ncols());
        for row in data.rows() {
            let diff = &row - &mean;
            std = std + &diff * &diff;
        }
        std = (std / n).mapv(f64::sqrt);

        // Avoid division by zero
        std = std.mapv(|v| if v.abs() < 1e-10 { 1.0 } else { v });

        self.mean = Some(mean);
        self.std = Some(std);
    }

    fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mean = self.mean.as_ref().expect("Normalizer not fitted");
        let std = self.std.as_ref().expect("Normalizer not fitted");

        let mut result = Array2::zeros(data.dim());
        for (i, row) in data.rows().into_iter().enumerate() {
            let normalized = (&row - mean) / std;
            result.row_mut(i).assign(&normalized);
        }
        result
    }

    fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let mean = self.mean.as_ref().expect("Normalizer not fitted");
        let std = self.std.as_ref().expect("Normalizer not fitted");

        let mut result = Array2::zeros(data.dim());
        for (i, row) in data.rows().into_iter().enumerate() {
            let denormalized = &row * std + mean;
            result.row_mut(i).assign(&denormalized);
        }
        result
    }
}

/// Normalize 1D array using min-max
pub fn normalize_1d(data: &[f64]) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }

    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    if range.abs() < 1e-10 {
        return vec![0.5; data.len()];
    }

    data.iter().map(|&x| (x - min) / range).collect()
}

/// Standardize 1D array (z-score)
pub fn standardize_1d(data: &[f64]) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }

    let n = data.len() as f64;
    let mean: f64 = data.iter().sum::<f64>() / n;
    let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    if std.abs() < 1e-10 {
        return vec![0.0; data.len()];
    }

    data.iter().map(|&x| (x - mean) / std).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minmax_normalizer() {
        let mut normalizer = MinMaxNormalizer::new();
        let data = Array2::from_shape_vec((4, 2), vec![0.0, 10.0, 5.0, 20.0, 10.0, 30.0, 15.0, 40.0])
            .unwrap();

        let normalized = normalizer.fit_transform(&data);

        // Check range [0, 1]
        assert!(normalized.iter().all(|&v| v >= 0.0 && v <= 1.0));

        // Check inverse
        let reconstructed = normalizer.inverse_transform(&normalized);
        for (a, b) in data.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_standard_normalizer() {
        let mut normalizer = StandardNormalizer::new();
        let data = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0])
            .unwrap();

        let normalized = normalizer.fit_transform(&data);

        // Check mean is approximately 0
        let mean = normalized.sum_axis(Axis(0)) / 4.0;
        assert!(mean.iter().all(|&v| v.abs() < 1e-10));

        // Check inverse
        let reconstructed = normalizer.inverse_transform(&normalized);
        for (a, b) in data.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_normalize_1d() {
        let data = vec![0.0, 50.0, 100.0];
        let normalized = normalize_1d(&data);
        assert_eq!(normalized, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_standardize_1d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let standardized = standardize_1d(&data);

        // Mean should be 0
        let mean: f64 = standardized.iter().sum::<f64>() / standardized.len() as f64;
        assert!(mean.abs() < 1e-10);
    }
}
