//! Data normalization utilities
//!
//! Provides functions for normalizing and standardizing data
//! for use in anomaly detection algorithms

use anyhow::{anyhow, Result};

/// Normalization methods
#[derive(Debug, Clone, Copy)]
pub enum NormMethod {
    /// Min-Max normalization to [0, 1]
    MinMax,
    /// Z-Score standardization (mean=0, std=1)
    ZScore,
    /// Robust scaling using median and IQR
    Robust,
    /// Percentile ranking [0, 1]
    Percentile,
}

/// Statistics for a data series
#[derive(Debug, Clone)]
pub struct DataStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub q1: f64,
    pub q3: f64,
}

impl DataStats {
    /// Calculate statistics from data
    pub fn from_data(data: &[f64]) -> Result<Self> {
        if data.is_empty() {
            return Err(anyhow!("Cannot calculate stats for empty data"));
        }

        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted[0];
        let max = sorted[sorted.len() - 1];

        let median = percentile(&sorted, 0.5);
        let q1 = percentile(&sorted, 0.25);
        let q3 = percentile(&sorted, 0.75);

        Ok(Self {
            mean,
            std,
            min,
            max,
            median,
            q1,
            q3,
        })
    }

    /// Get interquartile range (IQR)
    pub fn iqr(&self) -> f64 {
        self.q3 - self.q1
    }
}

/// Calculate percentile from sorted data
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let idx = p * (sorted.len() - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    let frac = idx - lower as f64;

    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

/// Normalize a single value using given method and stats
pub fn normalize_value(value: f64, method: NormMethod, stats: &DataStats) -> f64 {
    match method {
        NormMethod::MinMax => {
            if (stats.max - stats.min).abs() < 1e-10 {
                0.5
            } else {
                (value - stats.min) / (stats.max - stats.min)
            }
        }
        NormMethod::ZScore => {
            if stats.std < 1e-10 {
                0.0
            } else {
                (value - stats.mean) / stats.std
            }
        }
        NormMethod::Robust => {
            let iqr = stats.iqr();
            if iqr < 1e-10 {
                0.0
            } else {
                (value - stats.median) / iqr
            }
        }
        NormMethod::Percentile => {
            // Approximate percentile using normal distribution
            if stats.std < 1e-10 {
                0.5
            } else {
                let z = (value - stats.mean) / stats.std;
                // Approximate CDF using logistic function
                1.0 / (1.0 + (-1.702 * z).exp())
            }
        }
    }
}

/// Normalize an entire data series
pub fn normalize_series(data: &[f64], method: NormMethod) -> Result<Vec<f64>> {
    let stats = DataStats::from_data(data)?;
    Ok(data
        .iter()
        .map(|v| normalize_value(*v, method, &stats))
        .collect())
}

/// Rolling normalization with a window
pub fn rolling_normalize(data: &[f64], window: usize, method: NormMethod) -> Vec<f64> {
    if data.len() < window {
        return vec![0.0; data.len()];
    }

    let mut result = vec![0.0; window - 1];

    for i in (window - 1)..data.len() {
        let window_data: Vec<f64> = data[(i + 1 - window)..=i].to_vec();
        if let Ok(stats) = DataStats::from_data(&window_data) {
            result.push(normalize_value(data[i], method, &stats));
        } else {
            result.push(0.0);
        }
    }

    result
}

/// Multi-variate normalizer for feature matrices
pub struct FeatureNormalizer {
    method: NormMethod,
    feature_stats: Vec<DataStats>,
}

impl FeatureNormalizer {
    /// Fit normalizer on training data
    pub fn fit(features: &[Vec<f64>], method: NormMethod) -> Result<Self> {
        if features.is_empty() {
            return Err(anyhow!("No features provided"));
        }

        let n_features = features[0].len();
        let mut feature_stats = Vec::with_capacity(n_features);

        for f_idx in 0..n_features {
            let feature_values: Vec<f64> = features.iter().map(|row| row[f_idx]).collect();
            feature_stats.push(DataStats::from_data(&feature_values)?);
        }

        Ok(Self {
            method,
            feature_stats,
        })
    }

    /// Transform features using fitted parameters
    pub fn transform(&self, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        features
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(i, v)| {
                        if i < self.feature_stats.len() {
                            normalize_value(*v, self.method, &self.feature_stats[i])
                        } else {
                            *v
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Fit and transform in one step
    pub fn fit_transform(features: &[Vec<f64>], method: NormMethod) -> Result<Vec<Vec<f64>>> {
        let normalizer = Self::fit(features, method)?;
        Ok(normalizer.transform(features))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minmax_normalization() {
        let data = vec![0.0, 50.0, 100.0];
        let normalized = normalize_series(&data, NormMethod::MinMax).unwrap();

        assert!((normalized[0] - 0.0).abs() < 1e-6);
        assert!((normalized[1] - 0.5).abs() < 1e-6);
        assert!((normalized[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_zscore_normalization() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = normalize_series(&data, NormMethod::ZScore).unwrap();

        let mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert!(mean.abs() < 1e-6); // Mean should be ~0
    }

    #[test]
    fn test_data_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = DataStats::from_data(&data).unwrap();

        assert!((stats.mean - 3.0).abs() < 1e-6);
        assert!((stats.median - 3.0).abs() < 1e-6);
        assert!((stats.min - 1.0).abs() < 1e-6);
        assert!((stats.max - 5.0).abs() < 1e-6);
    }
}
