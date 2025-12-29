//! Data normalization utilities
//!
//! Functions for normalizing and standardizing data for anomaly detection

use ndarray::{Array1, Array2, Axis};

/// Normalizer for scaling data
pub struct Normalizer {
    pub method: NormalizationMethod,
    pub mean: Option<Array1<f64>>,
    pub std: Option<Array1<f64>>,
    pub min: Option<Array1<f64>>,
    pub max: Option<Array1<f64>>,
}

/// Normalization methods
#[derive(Clone, Copy, Debug)]
pub enum NormalizationMethod {
    /// Z-score normalization: (x - mean) / std
    ZScore,
    /// Min-Max normalization: (x - min) / (max - min)
    MinMax,
    /// Robust normalization: (x - median) / IQR
    Robust,
}

impl Normalizer {
    /// Create a new normalizer
    pub fn new(method: NormalizationMethod) -> Self {
        Self {
            method,
            mean: None,
            std: None,
            min: None,
            max: None,
        }
    }

    /// Fit the normalizer to data
    pub fn fit(&mut self, data: &Array2<f64>) {
        match self.method {
            NormalizationMethod::ZScore => {
                self.mean = Some(data.mean_axis(Axis(0)).unwrap());
                self.std = Some(data.std_axis(Axis(0), 0.0));
            }
            NormalizationMethod::MinMax => {
                self.min = Some(
                    data.fold_axis(Axis(0), f64::INFINITY, |&a, &b| a.min(b)),
                );
                self.max = Some(
                    data.fold_axis(Axis(0), f64::NEG_INFINITY, |&a, &b| a.max(b)),
                );
            }
            NormalizationMethod::Robust => {
                // For robust normalization, we compute median and IQR
                let mut medians = Vec::with_capacity(data.ncols());
                let mut iqrs = Vec::with_capacity(data.ncols());

                for col in data.columns() {
                    let mut sorted: Vec<f64> = col.iter().cloned().collect();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    let n = sorted.len();
                    let median = if n % 2 == 0 {
                        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                    } else {
                        sorted[n / 2]
                    };

                    let q1_idx = n / 4;
                    let q3_idx = 3 * n / 4;
                    let iqr = sorted[q3_idx] - sorted[q1_idx];

                    medians.push(median);
                    iqrs.push(iqr);
                }

                self.mean = Some(Array1::from(medians)); // Using mean field for median
                self.std = Some(Array1::from(iqrs)); // Using std field for IQR
            }
        }
    }

    /// Transform data using fitted parameters
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        match self.method {
            NormalizationMethod::ZScore => {
                let mean = self.mean.as_ref().expect("Normalizer not fitted");
                let std = self.std.as_ref().expect("Normalizer not fitted");

                let mut result = data.clone();
                for (i, mut col) in result.columns_mut().into_iter().enumerate() {
                    let s = if std[i] > 1e-10 { std[i] } else { 1.0 };
                    col.mapv_inplace(|x| (x - mean[i]) / s);
                }
                result
            }
            NormalizationMethod::MinMax => {
                let min = self.min.as_ref().expect("Normalizer not fitted");
                let max = self.max.as_ref().expect("Normalizer not fitted");

                let mut result = data.clone();
                for (i, mut col) in result.columns_mut().into_iter().enumerate() {
                    let range = max[i] - min[i];
                    let r = if range > 1e-10 { range } else { 1.0 };
                    col.mapv_inplace(|x| (x - min[i]) / r);
                }
                result
            }
            NormalizationMethod::Robust => {
                let median = self.mean.as_ref().expect("Normalizer not fitted");
                let iqr = self.std.as_ref().expect("Normalizer not fitted");

                let mut result = data.clone();
                for (i, mut col) in result.columns_mut().into_iter().enumerate() {
                    let r = if iqr[i] > 1e-10 { iqr[i] } else { 1.0 };
                    col.mapv_inplace(|x| (x - median[i]) / r);
                }
                result
            }
        }
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
        self.fit(data);
        self.transform(data)
    }

    /// Inverse transform (for MinMax and ZScore)
    pub fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        match self.method {
            NormalizationMethod::ZScore => {
                let mean = self.mean.as_ref().expect("Normalizer not fitted");
                let std = self.std.as_ref().expect("Normalizer not fitted");

                let mut result = data.clone();
                for (i, mut col) in result.columns_mut().into_iter().enumerate() {
                    col.mapv_inplace(|x| x * std[i] + mean[i]);
                }
                result
            }
            NormalizationMethod::MinMax => {
                let min = self.min.as_ref().expect("Normalizer not fitted");
                let max = self.max.as_ref().expect("Normalizer not fitted");

                let mut result = data.clone();
                for (i, mut col) in result.columns_mut().into_iter().enumerate() {
                    let range = max[i] - min[i];
                    col.mapv_inplace(|x| x * range + min[i]);
                }
                result
            }
            NormalizationMethod::Robust => {
                let median = self.mean.as_ref().expect("Normalizer not fitted");
                let iqr = self.std.as_ref().expect("Normalizer not fitted");

                let mut result = data.clone();
                for (i, mut col) in result.columns_mut().into_iter().enumerate() {
                    col.mapv_inplace(|x| x * iqr[i] + median[i]);
                }
                result
            }
        }
    }
}

/// Calculate rolling mean
pub fn rolling_mean(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || data.len() < window {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; window - 1];
    let mut sum: f64 = data[..window].iter().sum();
    result.push(sum / window as f64);

    for i in window..data.len() {
        sum += data[i] - data[i - window];
        result.push(sum / window as f64);
    }

    result
}

/// Calculate rolling standard deviation
pub fn rolling_std(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || data.len() < window {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; window - 1];

    for i in window - 1..data.len() {
        let slice = &data[i + 1 - window..=i];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
        result.push(variance.sqrt());
    }

    result
}

/// Calculate rolling median
pub fn rolling_median(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || data.len() < window {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; window - 1];

    for i in window - 1..data.len() {
        let mut slice: Vec<f64> = data[i + 1 - window..=i].to_vec();
        slice.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if window % 2 == 0 {
            (slice[window / 2 - 1] + slice[window / 2]) / 2.0
        } else {
            slice[window / 2]
        };

        result.push(median);
    }

    result
}

/// Calculate Median Absolute Deviation (MAD)
pub fn rolling_mad(data: &[f64], window: usize) -> Vec<f64> {
    let medians = rolling_median(data, window);

    if window == 0 || data.len() < window {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; window - 1];

    for i in window - 1..data.len() {
        let median = medians[i];
        let mut deviations: Vec<f64> = data[i + 1 - window..=i]
            .iter()
            .map(|x| (x - median).abs())
            .collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mad = if window % 2 == 0 {
            (deviations[window / 2 - 1] + deviations[window / 2]) / 2.0
        } else {
            deviations[window / 2]
        };

        result.push(mad);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_rolling_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_mean(&data, 3);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
        assert!((result[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_zscore_normalizer() {
        let data = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let mut normalizer = Normalizer::new(NormalizationMethod::ZScore);

        let normalized = normalizer.fit_transform(&data);
        let mean = normalized.mean_axis(Axis(0)).unwrap();

        assert!((mean[0]).abs() < 1e-10);
        assert!((mean[1]).abs() < 1e-10);
    }

    #[test]
    fn test_minmax_normalizer() {
        let data = array![[0.0, 0.0], [5.0, 10.0], [10.0, 20.0]];
        let mut normalizer = Normalizer::new(NormalizationMethod::MinMax);

        let normalized = normalizer.fit_transform(&data);

        assert!((normalized[[0, 0]] - 0.0).abs() < 1e-10);
        assert!((normalized[[2, 0]] - 1.0).abs() < 1e-10);
        assert!((normalized[[0, 1]] - 0.0).abs() < 1e-10);
        assert!((normalized[[2, 1]] - 1.0).abs() < 1e-10);
    }
}
