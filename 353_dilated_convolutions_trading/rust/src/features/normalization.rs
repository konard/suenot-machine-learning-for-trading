//! Data Normalization

use ndarray::Array2;

/// Data normalizer for preparing inputs to neural networks
#[derive(Debug, Clone)]
pub struct Normalizer {
    /// Means for each feature
    means: Vec<f64>,
    /// Standard deviations for each feature
    stds: Vec<f64>,
    /// Whether the normalizer has been fitted
    fitted: bool,
}

impl Default for Normalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Normalizer {
    /// Create a new normalizer
    pub fn new() -> Self {
        Self {
            means: Vec::new(),
            stds: Vec::new(),
            fitted: false,
        }
    }

    /// Create a normalizer with pre-computed statistics
    pub fn with_stats(means: Vec<f64>, stds: Vec<f64>) -> Self {
        assert_eq!(means.len(), stds.len());
        Self {
            means,
            stds,
            fitted: true,
        }
    }

    /// Fit the normalizer to data
    ///
    /// # Arguments
    /// - `data` - Data of shape (n_features, n_samples)
    pub fn fit(&mut self, data: &Array2<f64>) {
        let (n_features, n_samples) = data.dim();

        self.means = Vec::with_capacity(n_features);
        self.stds = Vec::with_capacity(n_features);

        for i in 0..n_features {
            let row = data.row(i);
            let mean: f64 = row.iter().sum::<f64>() / n_samples as f64;
            let variance: f64 = row.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;
            let std = variance.sqrt().max(1e-8); // Avoid division by zero

            self.means.push(mean);
            self.stds.push(std);
        }

        self.fitted = true;
    }

    /// Transform data using fitted statistics
    ///
    /// # Arguments
    /// - `data` - Data of shape (n_features, n_samples)
    ///
    /// # Returns
    /// - Normalized data of shape (n_features, n_samples)
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        assert!(self.fitted, "Normalizer must be fitted before transform");

        let (n_features, n_samples) = data.dim();
        assert_eq!(n_features, self.means.len());

        let mut normalized = Array2::zeros((n_features, n_samples));

        for i in 0..n_features {
            for j in 0..n_samples {
                normalized[[i, j]] = (data[[i, j]] - self.means[i]) / self.stds[i];
            }
        }

        normalized
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
        self.fit(data);
        self.transform(data)
    }

    /// Inverse transform to original scale
    pub fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        assert!(self.fitted, "Normalizer must be fitted before inverse_transform");

        let (n_features, n_samples) = data.dim();
        assert_eq!(n_features, self.means.len());

        let mut original = Array2::zeros((n_features, n_samples));

        for i in 0..n_features {
            for j in 0..n_samples {
                original[[i, j]] = data[[i, j]] * self.stds[i] + self.means[i];
            }
        }

        original
    }

    /// Get fitted means
    pub fn means(&self) -> &[f64] {
        &self.means
    }

    /// Get fitted standard deviations
    pub fn stds(&self) -> &[f64] {
        &self.stds
    }

    /// Check if fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }
}

/// Min-Max normalizer (scales to [0, 1])
#[derive(Debug, Clone)]
pub struct MinMaxNormalizer {
    /// Minimum values for each feature
    mins: Vec<f64>,
    /// Maximum values for each feature
    maxs: Vec<f64>,
    /// Whether fitted
    fitted: bool,
}

impl Default for MinMaxNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl MinMaxNormalizer {
    /// Create a new min-max normalizer
    pub fn new() -> Self {
        Self {
            mins: Vec::new(),
            maxs: Vec::new(),
            fitted: false,
        }
    }

    /// Fit to data
    pub fn fit(&mut self, data: &Array2<f64>) {
        let (n_features, _) = data.dim();

        self.mins = Vec::with_capacity(n_features);
        self.maxs = Vec::with_capacity(n_features);

        for i in 0..n_features {
            let row = data.row(i);
            let min = row.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            self.mins.push(min);
            self.maxs.push(if (max - min).abs() < 1e-8 { min + 1.0 } else { max });
        }

        self.fitted = true;
    }

    /// Transform data to [0, 1] range
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        assert!(self.fitted, "Normalizer must be fitted before transform");

        let (n_features, n_samples) = data.dim();
        let mut normalized = Array2::zeros((n_features, n_samples));

        for i in 0..n_features {
            let range = self.maxs[i] - self.mins[i];
            for j in 0..n_samples {
                normalized[[i, j]] = (data[[i, j]] - self.mins[i]) / range;
            }
        }

        normalized
    }

    /// Fit and transform
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
        self.fit(data);
        self.transform(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_normalizer_fit_transform() {
        let data = array![[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]];

        let mut normalizer = Normalizer::new();
        let normalized = normalizer.fit_transform(&data);

        // Mean should be ~0, std should be ~1
        for i in 0..2 {
            let row = normalized.row(i);
            let mean: f64 = row.iter().sum::<f64>() / row.len() as f64;
            assert!(mean.abs() < 1e-10);
        }
    }

    #[test]
    fn test_inverse_transform() {
        let data = array![[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]];

        let mut normalizer = Normalizer::new();
        let normalized = normalizer.fit_transform(&data);
        let recovered = normalizer.inverse_transform(&normalized);

        for i in 0..2 {
            for j in 0..3 {
                assert!((data[[i, j]] - recovered[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_minmax_normalizer() {
        let data = array![[0.0, 50.0, 100.0], [0.0, 5.0, 10.0]];

        let mut normalizer = MinMaxNormalizer::new();
        let normalized = normalizer.fit_transform(&data);

        assert!((normalized[[0, 0]] - 0.0).abs() < 1e-10);
        assert!((normalized[[0, 1]] - 0.5).abs() < 1e-10);
        assert!((normalized[[0, 2]] - 1.0).abs() < 1e-10);
    }
}
