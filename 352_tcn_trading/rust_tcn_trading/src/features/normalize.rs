//! Data Normalization Utilities

use ndarray::Array2;

/// Normalization method enumeration
#[derive(Debug, Clone, Copy)]
pub enum NormalizationMethod {
    /// Z-score normalization (mean=0, std=1)
    ZScore,
    /// Min-max normalization to [0, 1]
    MinMax,
    /// Robust scaling using median and IQR
    Robust,
    /// No normalization
    None,
}

/// Data normalizer
#[derive(Debug, Clone)]
pub struct Normalizer {
    /// Normalization method
    pub method: NormalizationMethod,
    /// Rolling window size (None for global normalization)
    pub window: Option<usize>,
    /// Stored statistics for inverse transform
    stats: Option<NormalizerStats>,
}

#[derive(Debug, Clone)]
struct NormalizerStats {
    means: Vec<f64>,
    stds: Vec<f64>,
    mins: Vec<f64>,
    maxs: Vec<f64>,
    medians: Vec<f64>,
    iqrs: Vec<f64>,
}

impl Default for Normalizer {
    fn default() -> Self {
        Self::new(NormalizationMethod::ZScore, None)
    }
}

impl Normalizer {
    /// Create a new normalizer
    pub fn new(method: NormalizationMethod, window: Option<usize>) -> Self {
        Self {
            method,
            window,
            stats: None,
        }
    }

    /// Create a Z-score normalizer
    pub fn zscore() -> Self {
        Self::new(NormalizationMethod::ZScore, None)
    }

    /// Create a Min-Max normalizer
    pub fn minmax() -> Self {
        Self::new(NormalizationMethod::MinMax, None)
    }

    /// Create a robust normalizer
    pub fn robust() -> Self {
        Self::new(NormalizationMethod::Robust, None)
    }

    /// Create a rolling window normalizer
    pub fn rolling(method: NormalizationMethod, window: usize) -> Self {
        Self::new(method, Some(window))
    }

    /// Fit the normalizer to data and transform
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
        let (num_features, seq_len) = data.dim();

        match self.window {
            Some(window) => self.rolling_normalize(data, window),
            None => {
                // Global normalization
                let mut stats = NormalizerStats {
                    means: Vec::with_capacity(num_features),
                    stds: Vec::with_capacity(num_features),
                    mins: Vec::with_capacity(num_features),
                    maxs: Vec::with_capacity(num_features),
                    medians: Vec::with_capacity(num_features),
                    iqrs: Vec::with_capacity(num_features),
                };

                let mut result = Array2::zeros((num_features, seq_len));

                for i in 0..num_features {
                    let row = data.row(i);
                    let values: Vec<f64> = row.iter().filter(|x| !x.is_nan()).cloned().collect();

                    if values.is_empty() {
                        stats.means.push(0.0);
                        stats.stds.push(1.0);
                        stats.mins.push(0.0);
                        stats.maxs.push(1.0);
                        stats.medians.push(0.0);
                        stats.iqrs.push(1.0);
                        continue;
                    }

                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
                    let std = variance.sqrt().max(1e-10);
                    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                    let mut sorted = values.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median = sorted[sorted.len() / 2];
                    let q1 = sorted[sorted.len() / 4];
                    let q3 = sorted[3 * sorted.len() / 4];
                    let iqr = (q3 - q1).max(1e-10);

                    stats.means.push(mean);
                    stats.stds.push(std);
                    stats.mins.push(min);
                    stats.maxs.push(max);
                    stats.medians.push(median);
                    stats.iqrs.push(iqr);

                    for j in 0..seq_len {
                        let val = data[[i, j]];
                        if val.is_nan() {
                            result[[i, j]] = 0.0;
                        } else {
                            result[[i, j]] = match self.method {
                                NormalizationMethod::ZScore => (val - mean) / std,
                                NormalizationMethod::MinMax => {
                                    if max - min > 1e-10 {
                                        (val - min) / (max - min)
                                    } else {
                                        0.5
                                    }
                                }
                                NormalizationMethod::Robust => (val - median) / iqr,
                                NormalizationMethod::None => val,
                            };
                        }
                    }
                }

                self.stats = Some(stats);
                result
            }
        }
    }

    /// Transform data using previously fitted statistics
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let (num_features, seq_len) = data.dim();

        match (&self.stats, self.window) {
            (Some(stats), None) => {
                let mut result = Array2::zeros((num_features, seq_len));

                for i in 0..num_features {
                    for j in 0..seq_len {
                        let val = data[[i, j]];
                        if val.is_nan() {
                            result[[i, j]] = 0.0;
                        } else {
                            result[[i, j]] = match self.method {
                                NormalizationMethod::ZScore => (val - stats.means[i]) / stats.stds[i],
                                NormalizationMethod::MinMax => {
                                    let range = stats.maxs[i] - stats.mins[i];
                                    if range > 1e-10 {
                                        (val - stats.mins[i]) / range
                                    } else {
                                        0.5
                                    }
                                }
                                NormalizationMethod::Robust => (val - stats.medians[i]) / stats.iqrs[i],
                                NormalizationMethod::None => val,
                            };
                        }
                    }
                }

                result
            }
            (_, Some(window)) => self.rolling_normalize(data, window),
            (None, None) => data.clone(),
        }
    }

    /// Rolling window normalization
    fn rolling_normalize(&self, data: &Array2<f64>, window: usize) -> Array2<f64> {
        let (num_features, seq_len) = data.dim();
        let mut result = Array2::zeros((num_features, seq_len));

        for i in 0..num_features {
            for j in 0..seq_len {
                let start = if j >= window { j - window + 1 } else { 0 };
                let slice: Vec<f64> = (start..=j)
                    .filter_map(|k| {
                        let v = data[[i, k]];
                        if v.is_nan() { None } else { Some(v) }
                    })
                    .collect();

                if slice.is_empty() {
                    result[[i, j]] = 0.0;
                    continue;
                }

                let val = data[[i, j]];
                if val.is_nan() {
                    result[[i, j]] = 0.0;
                    continue;
                }

                let mean = slice.iter().sum::<f64>() / slice.len() as f64;
                let variance = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / slice.len() as f64;
                let std = variance.sqrt().max(1e-10);
                let min = slice.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                result[[i, j]] = match self.method {
                    NormalizationMethod::ZScore => (val - mean) / std,
                    NormalizationMethod::MinMax => {
                        if max - min > 1e-10 {
                            (val - min) / (max - min)
                        } else {
                            0.5
                        }
                    }
                    NormalizationMethod::Robust => {
                        let mut sorted = slice.clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let median = sorted[sorted.len() / 2];
                        let q1 = sorted[sorted.len() / 4];
                        let q3 = sorted[3 * sorted.len() / 4];
                        let iqr = (q3 - q1).max(1e-10);
                        (val - median) / iqr
                    }
                    NormalizationMethod::None => val,
                };
            }
        }

        result
    }

    /// Clip values to a range
    pub fn clip(data: &Array2<f64>, min: f64, max: f64) -> Array2<f64> {
        data.mapv(|x| x.max(min).min(max))
    }

    /// Replace NaN values
    pub fn fill_nan(data: &Array2<f64>, fill_value: f64) -> Array2<f64> {
        data.mapv(|x| if x.is_nan() { fill_value } else { x })
    }

    /// Forward fill NaN values
    pub fn ffill(data: &Array2<f64>) -> Array2<f64> {
        let (num_features, seq_len) = data.dim();
        let mut result = data.clone();

        for i in 0..num_features {
            let mut last_valid = 0.0;
            for j in 0..seq_len {
                if result[[i, j]].is_nan() {
                    result[[i, j]] = last_valid;
                } else {
                    last_valid = result[[i, j]];
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
    fn test_zscore_normalization() {
        let data = Array2::from_shape_vec((1, 5), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let mut normalizer = Normalizer::zscore();
        let normalized = normalizer.fit_transform(&data);

        // Mean should be close to 0
        let mean: f64 = normalized.iter().sum::<f64>() / 5.0;
        assert!(mean.abs() < 0.001);

        // Std should be close to 1
        let std: f64 = (normalized.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 5.0).sqrt();
        assert!((std - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_minmax_normalization() {
        let data = Array2::from_shape_vec((1, 5), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let mut normalizer = Normalizer::minmax();
        let normalized = normalizer.fit_transform(&data);

        // Min should be 0, max should be 1
        let min = normalized.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = normalized.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(min.abs() < 0.001);
        assert!((max - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_rolling_normalization() {
        let data = Array2::from_shape_vec((1, 10), (1..=10).map(|x| x as f64).collect()).unwrap();
        let mut normalizer = Normalizer::rolling(NormalizationMethod::ZScore, 3);
        let normalized = normalizer.fit_transform(&data);

        assert_eq!(normalized.dim(), (1, 10));
    }

    #[test]
    fn test_fill_nan() {
        let data = Array2::from_shape_vec((1, 3), vec![1.0, f64::NAN, 3.0]).unwrap();
        let filled = Normalizer::fill_nan(&data, 0.0);
        assert_eq!(filled[[0, 1]], 0.0);
    }

    #[test]
    fn test_ffill() {
        let data = Array2::from_shape_vec((1, 4), vec![1.0, f64::NAN, f64::NAN, 4.0]).unwrap();
        let filled = Normalizer::ffill(&data);
        assert_eq!(filled[[0, 1]], 1.0);
        assert_eq!(filled[[0, 2]], 1.0);
        assert_eq!(filled[[0, 3]], 4.0);
    }
}
