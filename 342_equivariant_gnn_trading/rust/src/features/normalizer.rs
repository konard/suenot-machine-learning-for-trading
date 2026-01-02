//! Data Normalization

/// Min-max normalize values to [0, 1]
pub fn normalize(values: &[f64]) -> Vec<f64> {
    if values.is_empty() { return vec![]; }
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    if range.abs() < 1e-10 { return vec![0.5; values.len()]; }
    values.iter().map(|v| (v - min) / range).collect()
}

/// Standardize values to zero mean and unit variance
pub fn standardize(values: &[f64]) -> Vec<f64> {
    if values.len() < 2 { return vec![0.0; values.len()]; }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let std = (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64).sqrt();
    if std.abs() < 1e-10 { return vec![0.0; values.len()]; }
    values.iter().map(|v| (v - mean) / std).collect()
}

/// Normalizer that remembers statistics
pub struct Normalizer {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

impl Normalizer {
    /// Fit normalizer to data
    pub fn fit(values: &[f64]) -> Self {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let std = (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Self { mean, std, min, max }
    }

    /// Transform using fitted statistics
    pub fn transform(&self, values: &[f64]) -> Vec<f64> {
        if self.std.abs() < 1e-10 { return vec![0.0; values.len()]; }
        values.iter().map(|v| (v - self.mean) / self.std).collect()
    }

    /// Inverse transform
    pub fn inverse_transform(&self, values: &[f64]) -> Vec<f64> {
        values.iter().map(|v| v * self.std + self.mean).collect()
    }
}
