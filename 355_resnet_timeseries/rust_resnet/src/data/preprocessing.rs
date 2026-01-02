//! Data preprocessing utilities

use ndarray::{Array1, Array3, Axis};
use serde::{Deserialize, Serialize};

/// Standard scaler for z-score normalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardScaler {
    /// Mean for each feature channel
    pub mean: Array1<f32>,
    /// Standard deviation for each feature channel
    pub std: Array1<f32>,
    /// Number of features
    pub num_features: usize,
    /// Whether the scaler has been fitted
    pub fitted: bool,
}

impl StandardScaler {
    /// Create a new StandardScaler
    pub fn new(num_features: usize) -> Self {
        Self {
            mean: Array1::zeros(num_features),
            std: Array1::ones(num_features),
            num_features,
            fitted: false,
        }
    }

    /// Fit the scaler to training data
    ///
    /// # Arguments
    ///
    /// * `data` - Training data with shape [samples, features, sequence_length]
    pub fn fit(&mut self, data: &Array3<f32>) {
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];
        let seq_len = data.shape()[2];

        assert_eq!(n_features, self.num_features);

        let total_elements = (n_samples * seq_len) as f32;

        for f in 0..n_features {
            // Calculate mean
            let mut sum = 0.0f32;
            for s in 0..n_samples {
                for t in 0..seq_len {
                    sum += data[[s, f, t]];
                }
            }
            self.mean[f] = sum / total_elements;

            // Calculate std
            let mut sq_sum = 0.0f32;
            for s in 0..n_samples {
                for t in 0..seq_len {
                    let diff = data[[s, f, t]] - self.mean[f];
                    sq_sum += diff * diff;
                }
            }
            self.std[f] = (sq_sum / total_elements).sqrt().max(1e-8);
        }

        self.fitted = true;
    }

    /// Transform data using fitted parameters
    pub fn transform(&self, data: &Array3<f32>) -> Array3<f32> {
        assert!(self.fitted, "Scaler must be fitted before transform");

        let mut result = data.clone();
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];
        let seq_len = data.shape()[2];

        for s in 0..n_samples {
            for f in 0..n_features {
                for t in 0..seq_len {
                    result[[s, f, t]] = (data[[s, f, t]] - self.mean[f]) / self.std[f];
                }
            }
        }

        result
    }

    /// Fit and transform data
    pub fn fit_transform(&mut self, data: &Array3<f32>) -> Array3<f32> {
        self.fit(data);
        self.transform(data)
    }

    /// Inverse transform data
    pub fn inverse_transform(&self, data: &Array3<f32>) -> Array3<f32> {
        assert!(self.fitted, "Scaler must be fitted before inverse_transform");

        let mut result = data.clone();
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];
        let seq_len = data.shape()[2];

        for s in 0..n_samples {
            for f in 0..n_features {
                for t in 0..seq_len {
                    result[[s, f, t]] = data[[s, f, t]] * self.std[f] + self.mean[f];
                }
            }
        }

        result
    }
}

/// Generic normalizer trait
pub trait Normalizer {
    /// Fit to data
    fn fit(&mut self, data: &Array3<f32>);

    /// Transform data
    fn transform(&self, data: &Array3<f32>) -> Array3<f32>;

    /// Fit and transform
    fn fit_transform(&mut self, data: &Array3<f32>) -> Array3<f32> {
        self.fit(data);
        self.transform(data)
    }
}

impl Normalizer for StandardScaler {
    fn fit(&mut self, data: &Array3<f32>) {
        StandardScaler::fit(self, data);
    }

    fn transform(&self, data: &Array3<f32>) -> Array3<f32> {
        StandardScaler::transform(self, data)
    }
}

/// Min-Max scaler for 0-1 normalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinMaxScaler {
    /// Minimum values for each feature
    pub min: Array1<f32>,
    /// Maximum values for each feature
    pub max: Array1<f32>,
    /// Number of features
    pub num_features: usize,
    /// Whether the scaler has been fitted
    pub fitted: bool,
}

impl MinMaxScaler {
    /// Create a new MinMaxScaler
    pub fn new(num_features: usize) -> Self {
        Self {
            min: Array1::zeros(num_features),
            max: Array1::ones(num_features),
            num_features,
            fitted: false,
        }
    }

    /// Fit the scaler to training data
    pub fn fit(&mut self, data: &Array3<f32>) {
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];
        let seq_len = data.shape()[2];

        assert_eq!(n_features, self.num_features);

        for f in 0..n_features {
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;

            for s in 0..n_samples {
                for t in 0..seq_len {
                    let val = data[[s, f, t]];
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                }
            }

            self.min[f] = min_val;
            self.max[f] = if (max_val - min_val).abs() > 1e-8 {
                max_val
            } else {
                min_val + 1.0
            };
        }

        self.fitted = true;
    }

    /// Transform data to 0-1 range
    pub fn transform(&self, data: &Array3<f32>) -> Array3<f32> {
        assert!(self.fitted, "Scaler must be fitted before transform");

        let mut result = data.clone();
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];
        let seq_len = data.shape()[2];

        for s in 0..n_samples {
            for f in 0..n_features {
                let range = self.max[f] - self.min[f];
                for t in 0..seq_len {
                    result[[s, f, t]] = (data[[s, f, t]] - self.min[f]) / range;
                }
            }
        }

        result
    }
}

impl Normalizer for MinMaxScaler {
    fn fit(&mut self, data: &Array3<f32>) {
        MinMaxScaler::fit(self, data);
    }

    fn transform(&self, data: &Array3<f32>) -> Array3<f32> {
        MinMaxScaler::transform(self, data)
    }
}

/// Data augmentation for time series
pub struct DataAugmenter {
    /// Gaussian noise standard deviation
    pub noise_std: f32,
    /// Magnitude scaling range (min, max)
    pub scale_range: (f32, f32),
    /// Whether to apply augmentations
    pub enabled: bool,
}

impl Default for DataAugmenter {
    fn default() -> Self {
        Self {
            noise_std: 0.01,
            scale_range: (0.9, 1.1),
            enabled: true,
        }
    }
}

impl DataAugmenter {
    /// Create a new DataAugmenter
    pub fn new(noise_std: f32, scale_range: (f32, f32)) -> Self {
        Self {
            noise_std,
            scale_range,
            enabled: true,
        }
    }

    /// Apply augmentations to a batch
    pub fn augment(&self, data: &Array3<f32>) -> Array3<f32> {
        if !self.enabled {
            return data.clone();
        }

        use rand::Rng;
        use rand_distr::Normal;

        let mut rng = rand::thread_rng();
        let mut result = data.clone();

        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];
        let seq_len = data.shape()[2];

        // Add Gaussian noise
        let noise_dist = Normal::new(0.0, self.noise_std).unwrap();

        for s in 0..n_samples {
            // Random scale factor for this sample
            let scale = rng.gen_range(self.scale_range.0..self.scale_range.1);

            for f in 0..n_features {
                for t in 0..seq_len {
                    let noise: f32 = rng.sample(noise_dist);
                    result[[s, f, t]] = data[[s, f, t]] * scale + noise;
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_scaler() {
        let mut scaler = StandardScaler::new(3);
        let data = Array3::from_shape_fn((10, 3, 20), |(s, f, t)| {
            (s * 100 + f * 10 + t) as f32
        });

        let transformed = scaler.fit_transform(&data);

        // Check that mean is approximately 0 for each feature
        for f in 0..3 {
            let feature_mean: f32 = transformed
                .slice(ndarray::s![.., f, ..])
                .iter()
                .sum::<f32>()
                / (10.0 * 20.0);
            assert!(feature_mean.abs() < 0.1);
        }
    }

    #[test]
    fn test_minmax_scaler() {
        let mut scaler = MinMaxScaler::new(3);
        let data = Array3::from_shape_fn((10, 3, 20), |(s, f, t)| {
            (s * 100 + f * 10 + t) as f32
        });

        scaler.fit(&data);
        let transformed = scaler.transform(&data);

        // Check that values are in [0, 1] range
        for val in transformed.iter() {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_inverse_transform() {
        let mut scaler = StandardScaler::new(3);
        let data = Array3::from_shape_fn((5, 3, 10), |(s, f, t)| {
            (s * 10 + f * 5 + t) as f32
        });

        let transformed = scaler.fit_transform(&data);
        let reconstructed = scaler.inverse_transform(&transformed);

        // Check that inverse transform recovers original data
        for (orig, recon) in data.iter().zip(reconstructed.iter()) {
            assert!((orig - recon).abs() < 1e-5);
        }
    }

    #[test]
    fn test_data_augmenter() {
        let augmenter = DataAugmenter::default();
        let data = Array3::ones((5, 3, 10));

        let augmented = augmenter.augment(&data);

        // Augmented data should be different from original
        assert!(data.iter().zip(augmented.iter()).any(|(a, b)| (a - b).abs() > 1e-6));
    }
}
