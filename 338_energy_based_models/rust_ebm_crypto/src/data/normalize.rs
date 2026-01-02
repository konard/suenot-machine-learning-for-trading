//! Data normalization utilities for preprocessing

use ndarray::{Array1, Array2, Axis};

/// Standard scaler for normalizing data to zero mean and unit variance
#[derive(Debug, Clone)]
pub struct StandardScaler {
    /// Mean of each feature
    pub mean: Array1<f64>,
    /// Standard deviation of each feature
    pub std: Array1<f64>,
    /// Whether the scaler has been fitted
    pub fitted: bool,
}

impl StandardScaler {
    /// Create a new unfitted scaler
    pub fn new() -> Self {
        Self {
            mean: Array1::zeros(0),
            std: Array1::ones(0),
            fitted: false,
        }
    }

    /// Fit the scaler to data
    pub fn fit(&mut self, data: &Array2<f64>) {
        self.mean = data.mean_axis(Axis(0)).unwrap();
        self.std = data.std_axis(Axis(0), 0.0);

        // Avoid division by zero
        self.std.mapv_inplace(|v| if v < 1e-10 { 1.0 } else { v });

        self.fitted = true;
    }

    /// Transform data using fitted parameters
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        assert!(self.fitted, "Scaler must be fitted before transform");

        let mut result = data.clone();
        for (i, mut col) in result.columns_mut().into_iter().enumerate() {
            col.mapv_inplace(|v| (v - self.mean[i]) / self.std[i]);
        }
        result
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
        self.fit(data);
        self.transform(data)
    }

    /// Inverse transform to original scale
    pub fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        assert!(self.fitted, "Scaler must be fitted before inverse_transform");

        let mut result = data.clone();
        for (i, mut col) in result.columns_mut().into_iter().enumerate() {
            col.mapv_inplace(|v| v * self.std[i] + self.mean[i]);
        }
        result
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

/// Min-max scaler for normalizing data to [0, 1] range
#[derive(Debug, Clone)]
pub struct MinMaxScaler {
    /// Minimum of each feature
    pub min: Array1<f64>,
    /// Maximum of each feature
    pub max: Array1<f64>,
    /// Whether the scaler has been fitted
    pub fitted: bool,
}

impl MinMaxScaler {
    /// Create a new unfitted scaler
    pub fn new() -> Self {
        Self {
            min: Array1::zeros(0),
            max: Array1::ones(0),
            fitted: false,
        }
    }

    /// Fit the scaler to data
    pub fn fit(&mut self, data: &Array2<f64>) {
        let n_features = data.ncols();
        self.min = Array1::zeros(n_features);
        self.max = Array1::zeros(n_features);

        for (i, col) in data.columns().into_iter().enumerate() {
            self.min[i] = col.iter().cloned().fold(f64::INFINITY, f64::min);
            self.max[i] = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        }

        self.fitted = true;
    }

    /// Transform data using fitted parameters
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        assert!(self.fitted, "Scaler must be fitted before transform");

        let mut result = data.clone();
        for (i, mut col) in result.columns_mut().into_iter().enumerate() {
            let range = self.max[i] - self.min[i];
            let range = if range < 1e-10 { 1.0 } else { range };
            col.mapv_inplace(|v| (v - self.min[i]) / range);
        }
        result
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
        self.fit(data);
        self.transform(data)
    }

    /// Inverse transform to original scale
    pub fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        assert!(self.fitted, "Scaler must be fitted before inverse_transform");

        let mut result = data.clone();
        for (i, mut col) in result.columns_mut().into_iter().enumerate() {
            let range = self.max[i] - self.min[i];
            col.mapv_inplace(|v| v * range + self.min[i]);
        }
        result
    }
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}

/// Robust scaler using median and IQR (less sensitive to outliers)
#[derive(Debug, Clone)]
pub struct RobustScaler {
    /// Median of each feature
    pub median: Array1<f64>,
    /// IQR (Q3 - Q1) of each feature
    pub iqr: Array1<f64>,
    /// Whether the scaler has been fitted
    pub fitted: bool,
}

impl RobustScaler {
    /// Create a new unfitted scaler
    pub fn new() -> Self {
        Self {
            median: Array1::zeros(0),
            iqr: Array1::ones(0),
            fitted: false,
        }
    }

    /// Compute quantile of a sorted slice
    fn quantile(sorted: &[f64], q: f64) -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let pos = (sorted.len() - 1) as f64 * q;
        let lower = pos.floor() as usize;
        let upper = (lower + 1).min(sorted.len() - 1);
        let frac = pos - lower as f64;
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }

    /// Fit the scaler to data
    pub fn fit(&mut self, data: &Array2<f64>) {
        let n_features = data.ncols();
        self.median = Array1::zeros(n_features);
        self.iqr = Array1::zeros(n_features);

        for (i, col) in data.columns().into_iter().enumerate() {
            let mut sorted: Vec<f64> = col.iter().cloned().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

            self.median[i] = Self::quantile(&sorted, 0.5);
            let q1 = Self::quantile(&sorted, 0.25);
            let q3 = Self::quantile(&sorted, 0.75);
            self.iqr[i] = q3 - q1;

            // Avoid division by zero
            if self.iqr[i] < 1e-10 {
                self.iqr[i] = 1.0;
            }
        }

        self.fitted = true;
    }

    /// Transform data using fitted parameters
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        assert!(self.fitted, "Scaler must be fitted before transform");

        let mut result = data.clone();
        for (i, mut col) in result.columns_mut().into_iter().enumerate() {
            col.mapv_inplace(|v| (v - self.median[i]) / self.iqr[i]);
        }
        result
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
        self.fit(data);
        self.transform(data)
    }
}

impl Default for RobustScaler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_standard_scaler() {
        let data = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]];

        let mut scaler = StandardScaler::new();
        let transformed = scaler.fit_transform(&data);

        // Check mean is approximately 0
        let mean = transformed.mean_axis(Axis(0)).unwrap();
        assert_relative_eq!(mean[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(mean[1], 0.0, epsilon = 1e-10);

        // Check std is approximately 1
        let std = transformed.std_axis(Axis(0), 0.0);
        assert_relative_eq!(std[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(std[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_minmax_scaler() {
        let data = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];

        let mut scaler = MinMaxScaler::new();
        let transformed = scaler.fit_transform(&data);

        // Check min is 0 and max is 1
        for col in transformed.columns() {
            let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            assert_relative_eq!(min, 0.0, epsilon = 1e-10);
            assert_relative_eq!(max, 1.0, epsilon = 1e-10);
        }
    }
}
