//! Feature Normalizer
//!
//! Normalizes features for neural network input.

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Feature normalization method
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// Min-max scaling to [0, 1]
    MinMax,
    /// Z-score standardization (mean=0, std=1)
    ZScore,
    /// Robust scaling using median and IQR
    Robust,
}

/// Feature normalizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureNormalizer {
    /// Normalization method
    pub method: NormalizationMethod,

    /// Number of features
    pub num_features: usize,

    /// Feature means (for z-score)
    pub means: Vec<f64>,

    /// Feature standard deviations (for z-score)
    pub stds: Vec<f64>,

    /// Feature minimums (for min-max)
    pub mins: Vec<f64>,

    /// Feature maximums (for min-max)
    pub maxs: Vec<f64>,

    /// Feature medians (for robust)
    pub medians: Vec<f64>,

    /// Feature IQRs (for robust)
    pub iqrs: Vec<f64>,

    /// Whether the normalizer has been fitted
    pub fitted: bool,
}

impl FeatureNormalizer {
    /// Create a new normalizer with the specified method
    pub fn new(method: NormalizationMethod, num_features: usize) -> Self {
        Self {
            method,
            num_features,
            means: vec![0.0; num_features],
            stds: vec![1.0; num_features],
            mins: vec![0.0; num_features],
            maxs: vec![1.0; num_features],
            medians: vec![0.0; num_features],
            iqrs: vec![1.0; num_features],
            fitted: false,
        }
    }

    /// Create a z-score normalizer
    pub fn zscore(num_features: usize) -> Self {
        Self::new(NormalizationMethod::ZScore, num_features)
    }

    /// Create a min-max normalizer
    pub fn minmax(num_features: usize) -> Self {
        Self::new(NormalizationMethod::MinMax, num_features)
    }

    /// Create a robust normalizer
    pub fn robust(num_features: usize) -> Self {
        Self::new(NormalizationMethod::Robust, num_features)
    }

    /// Fit the normalizer to data
    ///
    /// data: [num_samples, num_features]
    pub fn fit(&mut self, data: &Array2<f64>) {
        let (_, num_features) = data.dim();
        assert_eq!(num_features, self.num_features, "Feature count mismatch");

        for f in 0..num_features {
            let column: Vec<f64> = data.column(f).iter().cloned().collect();
            let valid: Vec<f64> = column.into_iter().filter(|x| !x.is_nan()).collect();

            if valid.is_empty() {
                continue;
            }

            match self.method {
                NormalizationMethod::MinMax => {
                    self.mins[f] = valid.iter().cloned().fold(f64::INFINITY, f64::min);
                    self.maxs[f] = valid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                }
                NormalizationMethod::ZScore => {
                    let n = valid.len() as f64;
                    let mean = valid.iter().sum::<f64>() / n;
                    let variance = valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
                    self.means[f] = mean;
                    self.stds[f] = variance.sqrt().max(1e-8);
                }
                NormalizationMethod::Robust => {
                    let mut sorted = valid.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    let n = sorted.len();
                    self.medians[f] = if n % 2 == 0 {
                        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                    } else {
                        sorted[n / 2]
                    };

                    let q1 = sorted[n / 4];
                    let q3 = sorted[3 * n / 4];
                    self.iqrs[f] = (q3 - q1).max(1e-8);
                }
            }
        }

        self.fitted = true;
    }

    /// Transform data using fitted parameters
    ///
    /// data: [num_samples, num_features]
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        assert!(self.fitted, "Normalizer must be fitted first");

        let (num_samples, num_features) = data.dim();
        assert_eq!(num_features, self.num_features, "Feature count mismatch");

        let mut result = data.clone();

        for f in 0..num_features {
            for i in 0..num_samples {
                let val = data[[i, f]];
                if val.is_nan() {
                    result[[i, f]] = 0.0;
                    continue;
                }

                result[[i, f]] = match self.method {
                    NormalizationMethod::MinMax => {
                        let range = self.maxs[f] - self.mins[f];
                        if range > 0.0 {
                            ((val - self.mins[f]) / range).max(0.0).min(1.0)
                        } else {
                            0.5
                        }
                    }
                    NormalizationMethod::ZScore => {
                        (val - self.means[f]) / self.stds[f]
                    }
                    NormalizationMethod::Robust => {
                        (val - self.medians[f]) / self.iqrs[f]
                    }
                };
            }
        }

        result
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
        self.fit(data);
        self.transform(data)
    }

    /// Inverse transform (for z-score and min-max)
    pub fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        assert!(self.fitted, "Normalizer must be fitted first");

        let (num_samples, num_features) = data.dim();
        let mut result = data.clone();

        for f in 0..num_features.min(self.num_features) {
            for i in 0..num_samples {
                let val = data[[i, f]];

                result[[i, f]] = match self.method {
                    NormalizationMethod::MinMax => {
                        val * (self.maxs[f] - self.mins[f]) + self.mins[f]
                    }
                    NormalizationMethod::ZScore => {
                        val * self.stds[f] + self.means[f]
                    }
                    NormalizationMethod::Robust => {
                        val * self.iqrs[f] + self.medians[f]
                    }
                };
            }
        }

        result
    }

    /// Get statistics summary
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("FeatureNormalizer ({})\n", match self.method {
            NormalizationMethod::MinMax => "MinMax",
            NormalizationMethod::ZScore => "ZScore",
            NormalizationMethod::Robust => "Robust",
        }));
        s.push_str(&format!("Features: {}\n", self.num_features));
        s.push_str(&format!("Fitted: {}\n", self.fitted));
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_zscore_normalization() {
        let data = array![
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ];

        let mut normalizer = FeatureNormalizer::zscore(2);
        let normalized = normalizer.fit_transform(&data);

        // Check mean is approximately 0
        let mean_col0 = normalized.column(0).mean().unwrap();
        let mean_col1 = normalized.column(1).mean().unwrap();
        assert!(mean_col0.abs() < 1e-10);
        assert!(mean_col1.abs() < 1e-10);

        // Check std is approximately 1
        let std_col0 = normalized.column(0).std(0.0);
        let std_col1 = normalized.column(1).std(0.0);
        assert!((std_col0 - 1.0).abs() < 0.1);
        assert!((std_col1 - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_minmax_normalization() {
        let data = array![
            [0.0, 100.0],
            [25.0, 150.0],
            [50.0, 200.0],
            [75.0, 250.0],
            [100.0, 300.0],
        ];

        let mut normalizer = FeatureNormalizer::minmax(2);
        let normalized = normalizer.fit_transform(&data);

        // Check range is [0, 1]
        for col in 0..2 {
            let min = normalized.column(col).iter().cloned().fold(f64::INFINITY, f64::min);
            let max = normalized.column(col).iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            assert!((min - 0.0).abs() < 1e-10);
            assert!((max - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_inverse_transform() {
        let data = array![
            [10.0, 100.0],
            [20.0, 200.0],
            [30.0, 300.0],
        ];

        let mut normalizer = FeatureNormalizer::zscore(2);
        let normalized = normalizer.fit_transform(&data);
        let recovered = normalizer.inverse_transform(&normalized);

        for i in 0..3 {
            for j in 0..2 {
                assert!((data[[i, j]] - recovered[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_robust_normalization() {
        let data = array![
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [100.0, 1000.0], // Outliers
        ];

        let mut normalizer = FeatureNormalizer::robust(2);
        let normalized = normalizer.fit_transform(&data);

        // Median values should be normalized to approximately 0
        // The outliers should not dominate the scaling
        assert!(normalized[[2, 0]].abs() < normalized[[3, 0]].abs());
    }
}
