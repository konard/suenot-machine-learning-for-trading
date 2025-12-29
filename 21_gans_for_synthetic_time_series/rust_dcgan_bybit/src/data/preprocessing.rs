//! Data preprocessing utilities for GAN training
//!
//! This module provides functions for:
//! - Normalizing price data to [-1, 1] range (required for GAN with tanh activation)
//! - Creating sliding window sequences for time series
//! - Augmenting training data

use ndarray::{Array2, Array3, Axis};

/// Normalization parameters for denormalization
#[derive(Debug, Clone)]
pub struct NormalizationParams {
    pub min_vals: Vec<f64>,
    pub max_vals: Vec<f64>,
}

/// Normalize data to [-1, 1] range using min-max normalization
///
/// Formula: x_norm = 2 * (x - min) / (max - min) - 1
///
/// # Arguments
///
/// * `data` - 2D array of shape (num_samples, num_features)
///
/// # Returns
///
/// Tuple of (normalized data, normalization parameters)
pub fn normalize_data(data: &Array2<f64>) -> (Array2<f64>, NormalizationParams) {
    let num_features = data.ncols();
    let mut min_vals = vec![f64::MAX; num_features];
    let mut max_vals = vec![f64::MIN; num_features];

    // Find min and max for each feature
    for col in 0..num_features {
        let column = data.column(col);
        for &val in column.iter() {
            if val < min_vals[col] {
                min_vals[col] = val;
            }
            if val > max_vals[col] {
                max_vals[col] = val;
            }
        }
    }

    // Normalize to [-1, 1]
    let mut normalized = data.clone();
    for col in 0..num_features {
        let range = max_vals[col] - min_vals[col];
        if range > 0.0 {
            for row in 0..data.nrows() {
                normalized[[row, col]] =
                    2.0 * (data[[row, col]] - min_vals[col]) / range - 1.0;
            }
        } else {
            // If range is 0, set all values to 0
            for row in 0..data.nrows() {
                normalized[[row, col]] = 0.0;
            }
        }
    }

    let params = NormalizationParams { min_vals, max_vals };
    (normalized, params)
}

/// Denormalize data back to original scale
///
/// Formula: x = (x_norm + 1) / 2 * (max - min) + min
///
/// # Arguments
///
/// * `data` - Normalized data in [-1, 1] range
/// * `params` - Normalization parameters from normalize_data
///
/// # Returns
///
/// Denormalized data in original scale
pub fn denormalize_data(data: &Array2<f64>, params: &NormalizationParams) -> Array2<f64> {
    let num_features = data.ncols();
    let mut denormalized = data.clone();

    for col in 0..num_features {
        let range = params.max_vals[col] - params.min_vals[col];
        for row in 0..data.nrows() {
            denormalized[[row, col]] =
                (data[[row, col]] + 1.0) / 2.0 * range + params.min_vals[col];
        }
    }

    denormalized
}

/// Create sliding window sequences from time series data
///
/// # Arguments
///
/// * `data` - 2D array of shape (num_samples, num_features)
/// * `sequence_length` - Length of each sequence window
/// * `step` - Step size between sequences (1 for overlapping)
///
/// # Returns
///
/// 3D array of shape (num_sequences, sequence_length, num_features)
pub fn create_sequences(
    data: &Array2<f64>,
    sequence_length: usize,
    step: usize,
) -> Array3<f64> {
    let num_samples = data.nrows();
    let num_features = data.ncols();

    if num_samples < sequence_length {
        panic!(
            "Data length ({}) must be >= sequence_length ({})",
            num_samples, sequence_length
        );
    }

    // Calculate number of sequences
    let num_sequences = (num_samples - sequence_length) / step + 1;

    // Create output array
    let mut sequences = Array3::<f64>::zeros((num_sequences, sequence_length, num_features));

    for (seq_idx, start) in (0..=num_samples - sequence_length).step_by(step).enumerate() {
        for t in 0..sequence_length {
            for f in 0..num_features {
                sequences[[seq_idx, t, f]] = data[[start + t, f]];
            }
        }
    }

    sequences
}

/// Calculate log returns from price data
///
/// # Arguments
///
/// * `prices` - 1D slice of prices
///
/// # Returns
///
/// Vector of log returns (length = prices.len() - 1)
pub fn calculate_log_returns(prices: &[f64]) -> Vec<f64> {
    prices
        .windows(2)
        .map(|w| {
            if w[0] > 0.0 && w[1] > 0.0 {
                (w[1] / w[0]).ln()
            } else {
                0.0
            }
        })
        .collect()
}

/// Convert log returns back to prices
///
/// # Arguments
///
/// * `log_returns` - Vector of log returns
/// * `initial_price` - Starting price
///
/// # Returns
///
/// Vector of reconstructed prices
pub fn log_returns_to_prices(log_returns: &[f64], initial_price: f64) -> Vec<f64> {
    let mut prices = vec![initial_price];

    for &ret in log_returns {
        let last_price = *prices.last().unwrap();
        prices.push(last_price * ret.exp());
    }

    prices
}

/// Add noise to data for augmentation
///
/// # Arguments
///
/// * `data` - Input data
/// * `noise_level` - Standard deviation of Gaussian noise
///
/// # Returns
///
/// Augmented data with added noise
pub fn add_gaussian_noise(data: &Array2<f64>, noise_level: f64) -> Array2<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut augmented = data.clone();
    for val in augmented.iter_mut() {
        let noise: f64 = rng.gen::<f64>() * 2.0 - 1.0; // Uniform [-1, 1]
        *val += noise * noise_level;
        // Clip to [-1, 1] if normalized
        *val = val.clamp(-1.0, 1.0);
    }

    augmented
}

/// Shuffle sequences along the first axis
pub fn shuffle_sequences(data: &mut Array3<f64>) {
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();

    let num_sequences = data.shape()[0];
    let mut indices: Vec<usize> = (0..num_sequences).collect();
    indices.shuffle(&mut rng);

    let original = data.clone();
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        data.index_axis_mut(Axis(0), new_idx)
            .assign(&original.index_axis(Axis(0), old_idx));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_normalize_denormalize() {
        let data = array![[0.0, 100.0], [50.0, 200.0], [100.0, 300.0]];

        let (normalized, params) = normalize_data(&data);

        // Check normalized range is [-1, 1]
        for val in normalized.iter() {
            assert!(*val >= -1.0 && *val <= 1.0);
        }

        // Check denormalization recovers original
        let denormalized = denormalize_data(&normalized, &params);
        for (orig, denorm) in data.iter().zip(denormalized.iter()) {
            assert!((orig - denorm).abs() < 1e-10);
        }
    }

    #[test]
    fn test_create_sequences() {
        let data = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0]
        ];

        let sequences = create_sequences(&data, 3, 1);

        assert_eq!(sequences.shape(), &[3, 3, 2]);
        assert_eq!(sequences[[0, 0, 0]], 1.0);
        assert_eq!(sequences[[0, 2, 1]], 6.0);
        assert_eq!(sequences[[2, 0, 0]], 5.0);
    }

    #[test]
    fn test_log_returns() {
        let prices = vec![100.0, 105.0, 103.0, 108.0];
        let returns = calculate_log_returns(&prices);

        assert_eq!(returns.len(), 3);
        assert!((returns[0] - (105.0_f64 / 100.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_returns_to_prices() {
        let prices = vec![100.0, 105.0, 103.0, 108.0];
        let returns = calculate_log_returns(&prices);
        let reconstructed = log_returns_to_prices(&returns, 100.0);

        for (orig, recon) in prices.iter().zip(reconstructed.iter()) {
            assert!((orig - recon).abs() < 1e-10);
        }
    }
}
