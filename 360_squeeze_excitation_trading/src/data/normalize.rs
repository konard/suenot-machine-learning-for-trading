//! Data Normalization Utilities
//!
//! This module provides normalization methods for preparing data
//! for the SE trading model.

use ndarray::{Array1, Array2, Axis};

/// Normalization method to use
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormMethod {
    /// Z-score normalization (mean=0, std=1)
    ZScore,
    /// Min-Max normalization to [0, 1]
    MinMax,
    /// Min-Max normalization to [-1, 1]
    MinMaxSymmetric,
    /// Robust scaling using median and IQR
    Robust,
    /// No normalization
    None,
}

/// Statistics for normalization
#[derive(Debug, Clone)]
pub struct NormStats {
    pub mean: Array1<f64>,
    pub std: Array1<f64>,
    pub min: Array1<f64>,
    pub max: Array1<f64>,
    pub median: Array1<f64>,
    pub iqr: Array1<f64>,
}

impl NormStats {
    /// Compute statistics from data
    pub fn from_data(data: &Array2<f64>) -> Self {
        let n_features = data.ncols();

        let mean = data.mean_axis(Axis(0)).unwrap();

        let mut std = Array1::zeros(n_features);
        let mut min = Array1::from_elem(n_features, f64::INFINITY);
        let mut max = Array1::from_elem(n_features, f64::NEG_INFINITY);
        let mut median = Array1::zeros(n_features);
        let mut iqr = Array1::zeros(n_features);

        for j in 0..n_features {
            let col: Vec<f64> = data.column(j).to_vec();

            // Std
            let variance: f64 = col.iter().map(|x| (x - mean[j]).powi(2)).sum::<f64>()
                / col.len() as f64;
            std[j] = variance.sqrt().max(1e-8);

            // Min/Max
            for &v in &col {
                min[j] = min[j].min(v);
                max[j] = max[j].max(v);
            }

            // Median and IQR
            let mut sorted = col.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let n = sorted.len();
            median[j] = if n % 2 == 0 {
                (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
            } else {
                sorted[n / 2]
            };

            let q1_idx = n / 4;
            let q3_idx = 3 * n / 4;
            let q1 = sorted[q1_idx];
            let q3 = sorted[q3_idx];
            iqr[j] = (q3 - q1).max(1e-8);
        }

        Self {
            mean,
            std,
            min,
            max,
            median,
            iqr,
        }
    }
}

/// Data normalizer
#[derive(Debug, Clone)]
pub struct Normalizer {
    /// Normalization method
    method: NormMethod,
    /// Statistics (computed during fit)
    stats: Option<NormStats>,
    /// Epsilon for numerical stability
    eps: f64,
}

impl Normalizer {
    /// Create a new normalizer with the specified method
    pub fn new(method: NormMethod) -> Self {
        Self {
            method,
            stats: None,
            eps: 1e-8,
        }
    }

    /// Fit the normalizer to data (compute statistics)
    pub fn fit(&mut self, data: &Array2<f64>) {
        self.stats = Some(NormStats::from_data(data));
    }

    /// Transform data using fitted statistics
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        match self.method {
            NormMethod::None => data.clone(),
            _ => {
                let stats = self.stats.as_ref().expect("Normalizer not fitted");
                self.apply_normalization(data, stats)
            }
        }
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
        self.fit(data);
        self.transform(data)
    }

    /// Apply normalization based on method
    fn apply_normalization(&self, data: &Array2<f64>, stats: &NormStats) -> Array2<f64> {
        let mut result = data.clone();
        let n_features = data.ncols();

        for j in 0..n_features {
            match self.method {
                NormMethod::ZScore => {
                    let mean = stats.mean[j];
                    let std = stats.std[j];
                    result.column_mut(j).mapv_inplace(|x| (x - mean) / std);
                }
                NormMethod::MinMax => {
                    let min = stats.min[j];
                    let max = stats.max[j];
                    let range = (max - min).max(self.eps);
                    result.column_mut(j).mapv_inplace(|x| (x - min) / range);
                }
                NormMethod::MinMaxSymmetric => {
                    let min = stats.min[j];
                    let max = stats.max[j];
                    let range = (max - min).max(self.eps);
                    result
                        .column_mut(j)
                        .mapv_inplace(|x| 2.0 * (x - min) / range - 1.0);
                }
                NormMethod::Robust => {
                    let median = stats.median[j];
                    let iqr = stats.iqr[j];
                    result.column_mut(j).mapv_inplace(|x| (x - median) / iqr);
                }
                NormMethod::None => {}
            }
        }

        result
    }

    /// Inverse transform normalized data back to original scale
    pub fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        match self.method {
            NormMethod::None => data.clone(),
            _ => {
                let stats = self.stats.as_ref().expect("Normalizer not fitted");
                self.apply_inverse(data, stats)
            }
        }
    }

    /// Apply inverse normalization
    fn apply_inverse(&self, data: &Array2<f64>, stats: &NormStats) -> Array2<f64> {
        let mut result = data.clone();
        let n_features = data.ncols();

        for j in 0..n_features {
            match self.method {
                NormMethod::ZScore => {
                    let mean = stats.mean[j];
                    let std = stats.std[j];
                    result.column_mut(j).mapv_inplace(|x| x * std + mean);
                }
                NormMethod::MinMax => {
                    let min = stats.min[j];
                    let max = stats.max[j];
                    let range = max - min;
                    result.column_mut(j).mapv_inplace(|x| x * range + min);
                }
                NormMethod::MinMaxSymmetric => {
                    let min = stats.min[j];
                    let max = stats.max[j];
                    let range = max - min;
                    result
                        .column_mut(j)
                        .mapv_inplace(|x| (x + 1.0) / 2.0 * range + min);
                }
                NormMethod::Robust => {
                    let median = stats.median[j];
                    let iqr = stats.iqr[j];
                    result.column_mut(j).mapv_inplace(|x| x * iqr + median);
                }
                NormMethod::None => {}
            }
        }

        result
    }

    /// Get the normalization statistics
    pub fn stats(&self) -> Option<&NormStats> {
        self.stats.as_ref()
    }
}

impl Default for Normalizer {
    fn default() -> Self {
        Self::new(NormMethod::ZScore)
    }
}

/// Online normalizer for streaming data
#[derive(Debug, Clone)]
pub struct OnlineNormalizer {
    /// Running mean
    mean: Array1<f64>,
    /// Running variance (for std)
    var: Array1<f64>,
    /// Number of samples seen
    count: usize,
    /// Number of features
    n_features: usize,
}

impl OnlineNormalizer {
    /// Create a new online normalizer
    pub fn new(n_features: usize) -> Self {
        Self {
            mean: Array1::zeros(n_features),
            var: Array1::zeros(n_features),
            count: 0,
            n_features,
        }
    }

    /// Update statistics with new sample (Welford's algorithm)
    pub fn update(&mut self, sample: &Array1<f64>) {
        self.count += 1;
        let delta = sample - &self.mean;
        self.mean = &self.mean + &delta / self.count as f64;
        let delta2 = sample - &self.mean;
        self.var = &self.var + &(&delta * &delta2);
    }

    /// Get current standard deviation
    pub fn std(&self) -> Array1<f64> {
        if self.count < 2 {
            Array1::ones(self.n_features)
        } else {
            (&self.var / (self.count - 1) as f64).mapv(|x| x.sqrt().max(1e-8))
        }
    }

    /// Normalize a sample using current statistics
    pub fn normalize(&self, sample: &Array1<f64>) -> Array1<f64> {
        let std = self.std();
        (sample - &self.mean) / &std
    }

    /// Normalize a batch of samples
    pub fn normalize_batch(&self, data: &Array2<f64>) -> Array2<f64> {
        let std = self.std();
        let mut result = data.clone();

        for j in 0..self.n_features {
            result
                .column_mut(j)
                .mapv_inplace(|x| (x - self.mean[j]) / std[j]);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zscore_normalization() {
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0],
        )
        .unwrap();

        let mut normalizer = Normalizer::new(NormMethod::ZScore);
        let normalized = normalizer.fit_transform(&data);

        // Check mean is ~0 and std is ~1
        let mean = normalized.mean_axis(Axis(0)).unwrap();
        assert!(mean[0].abs() < 1e-10);
        assert!(mean[1].abs() < 1e-10);
    }

    #[test]
    fn test_minmax_normalization() {
        let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
            .unwrap();

        let mut normalizer = Normalizer::new(NormMethod::MinMax);
        let normalized = normalizer.fit_transform(&data);

        // Check values are in [0, 1]
        for val in normalized.iter() {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_inverse_transform() {
        let data = Array2::from_shape_vec((4, 2), vec![1.0, 100.0, 2.0, 200.0, 3.0, 300.0, 4.0, 400.0])
            .unwrap();

        let mut normalizer = Normalizer::new(NormMethod::ZScore);
        let normalized = normalizer.fit_transform(&data);
        let reconstructed = normalizer.inverse_transform(&normalized);

        for (a, b) in data.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_online_normalizer() {
        let mut online = OnlineNormalizer::new(2);

        // Add samples
        online.update(&Array1::from_vec(vec![1.0, 10.0]));
        online.update(&Array1::from_vec(vec![2.0, 20.0]));
        online.update(&Array1::from_vec(vec![3.0, 30.0]));

        // Check mean
        assert!((online.mean[0] - 2.0).abs() < 1e-10);
        assert!((online.mean[1] - 20.0).abs() < 1e-10);
    }
}
