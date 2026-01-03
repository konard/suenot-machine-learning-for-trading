//! Data preprocessing utilities
//!
//! Provides normalization and denormalization for model inputs

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Normalizer for z-score normalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Normalizer {
    /// Mean for each feature
    pub mean: Array1<f64>,
    /// Standard deviation for each feature
    pub std: Array1<f64>,
    /// Small epsilon to prevent division by zero
    pub epsilon: f64,
}

impl Normalizer {
    /// Fit normalizer to data
    pub fn fit(data: &Array2<f64>) -> Self {
        let mean = data.mean_axis(Axis(0)).expect("Failed to compute mean");
        let std = Self::compute_std(data, &mean);

        Self {
            mean,
            std,
            epsilon: 1e-8,
        }
    }

    /// Compute standard deviation along axis 0
    fn compute_std(data: &Array2<f64>, mean: &Array1<f64>) -> Array1<f64> {
        let n_samples = data.nrows() as f64;
        let centered = data - mean;
        let squared = &centered * &centered;
        let variance = squared.mean_axis(Axis(0)).expect("Failed to compute variance");
        variance.mapv(f64::sqrt)
    }

    /// Transform data using fitted parameters
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let centered = data - &self.mean;
        let std_safe = self.std.mapv(|s| if s < self.epsilon { 1.0 } else { s });
        centered / &std_safe
    }

    /// Inverse transform normalized data back to original scale
    pub fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let std_safe = self.std.mapv(|s| if s < self.epsilon { 1.0 } else { s });
        data * &std_safe + &self.mean
    }

    /// Transform a single sample
    pub fn transform_sample(&self, sample: &Array1<f64>) -> Array1<f64> {
        let centered = sample - &self.mean;
        let std_safe = self.std.mapv(|s| if s < self.epsilon { 1.0 } else { s });
        centered / &std_safe
    }

    /// Inverse transform a single sample
    pub fn inverse_transform_sample(&self, sample: &Array1<f64>) -> Array1<f64> {
        let std_safe = self.std.mapv(|s| if s < self.epsilon { 1.0 } else { s });
        sample * &std_safe + &self.mean
    }

    /// Clip values to a range (in standard deviations)
    pub fn clip_outliers(&self, data: &Array2<f64>, num_std: f64) -> Array2<f64> {
        let normalized = self.transform(data);
        let clipped = normalized.mapv(|v| v.clamp(-num_std, num_std));
        self.inverse_transform(&clipped)
    }
}

/// Normalize features using z-score normalization
pub fn normalize_features(data: &Array2<f64>) -> (Array2<f64>, Normalizer) {
    let normalizer = Normalizer::fit(data);
    let normalized = normalizer.transform(data);
    (normalized, normalizer)
}

/// Denormalize features back to original scale
pub fn denormalize_features(data: &Array2<f64>, normalizer: &Normalizer) -> Array2<f64> {
    normalizer.inverse_transform(data)
}

/// Split data into train, validation, and test sets
pub fn train_val_test_split(
    data: &Array2<f64>,
    train_ratio: f64,
    val_ratio: f64,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let n = data.nrows();
    let train_end = (n as f64 * train_ratio) as usize;
    let val_end = (n as f64 * (train_ratio + val_ratio)) as usize;

    let train = data.slice(ndarray::s![..train_end, ..]).to_owned();
    let val = data.slice(ndarray::s![train_end..val_end, ..]).to_owned();
    let test = data.slice(ndarray::s![val_end.., ..]).to_owned();

    (train, val, test)
}

/// Create sliding windows of data
pub fn create_windows(data: &Array2<f64>, window_size: usize) -> Vec<Array2<f64>> {
    let n = data.nrows();
    if n < window_size {
        return vec![];
    }

    (0..=n - window_size)
        .map(|i| data.slice(ndarray::s![i..i + window_size, ..]).to_owned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_normalizer() {
        let data = array![
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ];

        let normalizer = Normalizer::fit(&data);

        // Check mean
        assert!((normalizer.mean[0] - 3.0).abs() < 1e-10);
        assert!((normalizer.mean[1] - 30.0).abs() < 1e-10);

        // Transform and inverse transform
        let normalized = normalizer.transform(&data);
        let recovered = normalizer.inverse_transform(&normalized);

        for i in 0..data.nrows() {
            for j in 0..data.ncols() {
                assert!((data[[i, j]] - recovered[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_train_val_test_split() {
        let data = Array2::from_shape_vec((100, 5), (0..500).map(|x| x as f64).collect())
            .expect("Failed to create test data");

        let (train, val, test) = train_val_test_split(&data, 0.7, 0.15);

        assert_eq!(train.nrows(), 70);
        assert_eq!(val.nrows(), 15);
        assert_eq!(test.nrows(), 15);
    }

    #[test]
    fn test_create_windows() {
        let data = Array2::from_shape_vec((10, 3), (0..30).map(|x| x as f64).collect())
            .expect("Failed to create test data");

        let windows = create_windows(&data, 5);

        assert_eq!(windows.len(), 6);
        assert_eq!(windows[0].nrows(), 5);
        assert_eq!(windows[0].ncols(), 3);
    }
}
