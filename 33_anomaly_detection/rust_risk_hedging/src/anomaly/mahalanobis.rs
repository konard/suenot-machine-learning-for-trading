//! Mahalanobis Distance based anomaly detection
//!
//! Uses statistical distance accounting for correlations
//! between features to detect anomalies

use super::AnomalyDetector;
use anyhow::{anyhow, Result};

/// Mahalanobis distance based anomaly detector
#[derive(Debug, Clone)]
pub struct MahalanobisDetector {
    /// Mean vector
    mean: Vec<f64>,
    /// Inverse covariance matrix (flattened)
    cov_inv: Vec<Vec<f64>>,
    /// Threshold percentile (chi-squared distribution)
    threshold_percentile: f64,
    /// Degrees of freedom
    dof: usize,
}

impl MahalanobisDetector {
    /// Create detector from pre-computed parameters
    pub fn new(mean: Vec<f64>, cov_inv: Vec<Vec<f64>>, threshold_percentile: f64) -> Self {
        let dof = mean.len();
        Self {
            mean,
            cov_inv,
            threshold_percentile,
            dof,
        }
    }

    /// Fit detector on training data
    pub fn fit(data: &[Vec<f64>], threshold_percentile: f64) -> Result<Self> {
        if data.is_empty() {
            return Err(anyhow!("Empty data"));
        }

        let n_features = data[0].len();
        let n_samples = data.len() as f64;

        // Calculate mean
        let mut mean = vec![0.0; n_features];
        for row in data {
            for (i, v) in row.iter().enumerate() {
                mean[i] += v;
            }
        }
        for m in &mut mean {
            *m /= n_samples;
        }

        // Calculate covariance matrix
        let mut cov = vec![vec![0.0; n_features]; n_features];
        for row in data {
            for i in 0..n_features {
                for j in 0..n_features {
                    cov[i][j] += (row[i] - mean[i]) * (row[j] - mean[j]);
                }
            }
        }
        for i in 0..n_features {
            for j in 0..n_features {
                cov[i][j] /= n_samples - 1.0;
            }
        }

        // Add small regularization to diagonal
        for i in 0..n_features {
            cov[i][i] += 1e-6;
        }

        // Invert covariance matrix (using simple Gauss-Jordan for small matrices)
        let cov_inv = Self::invert_matrix(&cov)?;

        Ok(Self::new(mean, cov_inv, threshold_percentile))
    }

    /// Simple matrix inversion using Gauss-Jordan elimination
    fn invert_matrix(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n = matrix.len();
        if n == 0 {
            return Err(anyhow!("Empty matrix"));
        }

        // Create augmented matrix [A | I]
        let mut aug = vec![vec![0.0; 2 * n]; n];
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = matrix[i][j];
            }
            aug[i][n + i] = 1.0;
        }

        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            for row in (col + 1)..n {
                if aug[row][col].abs() > aug[max_row][col].abs() {
                    max_row = row;
                }
            }
            aug.swap(col, max_row);

            if aug[col][col].abs() < 1e-10 {
                return Err(anyhow!("Matrix is singular"));
            }

            // Scale pivot row
            let pivot = aug[col][col];
            for j in 0..(2 * n) {
                aug[col][j] /= pivot;
            }

            // Eliminate column
            for row in 0..n {
                if row != col {
                    let factor = aug[row][col];
                    for j in 0..(2 * n) {
                        aug[row][j] -= factor * aug[col][j];
                    }
                }
            }
        }

        // Extract inverse
        let mut inv = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                inv[i][j] = aug[i][n + j];
            }
        }

        Ok(inv)
    }

    /// Calculate Mahalanobis distance for a sample
    pub fn mahalanobis_distance(&self, sample: &[f64]) -> f64 {
        if sample.len() != self.mean.len() {
            return 0.0;
        }

        let n = self.mean.len();

        // (x - μ)
        let diff: Vec<f64> = sample.iter().zip(&self.mean).map(|(x, m)| x - m).collect();

        // (x - μ)ᵀ Σ⁻¹ (x - μ)
        let mut result = 0.0;
        for i in 0..n {
            for j in 0..n {
                result += diff[i] * self.cov_inv[i][j] * diff[j];
            }
        }

        result.sqrt()
    }

    /// Convert distance to anomaly score (0-1)
    pub fn distance_to_score(&self, distance: f64) -> f64 {
        // Use chi-squared CDF approximation
        // For d dimensions, D² follows chi-squared(d)
        let d_squared = distance * distance;

        // Approximate p-value using Wilson-Hilferty transformation
        let k = self.dof as f64;
        if k < 1.0 {
            return 0.5;
        }

        let x = d_squared / k;
        let z = (x.powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / (2.0 / (9.0 * k)).sqrt();

        // Convert z-score to probability using logistic approximation
        let p = 1.0 / (1.0 + (-1.702 * z).exp());

        p.clamp(0.0, 1.0)
    }

    /// Detect anomalies in multi-dimensional data
    pub fn detect_multi(&self, data: &[Vec<f64>]) -> Vec<f64> {
        data.iter()
            .map(|sample| {
                let dist = self.mahalanobis_distance(sample);
                self.distance_to_score(dist)
            })
            .collect()
    }
}

/// 1D Mahalanobis-like detector using rolling windows
#[derive(Debug, Clone)]
pub struct RollingMahalanobisDetector {
    window: usize,
    threshold_sigma: f64,
}

impl RollingMahalanobisDetector {
    /// Create new detector
    pub fn new(window: usize, threshold_sigma: f64) -> Self {
        Self {
            window: window.max(10),
            threshold_sigma: threshold_sigma.abs(),
        }
    }

    /// Calculate distance from rolling statistics
    fn rolling_distance(&self, value: f64, window_data: &[f64]) -> f64 {
        if window_data.is_empty() {
            return 0.0;
        }

        let n = window_data.len() as f64;
        let mean = window_data.iter().sum::<f64>() / n;
        let variance = window_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;

        if variance < 1e-10 {
            return 0.0;
        }

        // Mahalanobis distance in 1D = |x - μ| / σ
        ((value - mean).powi(2) / variance).sqrt()
    }

    /// Convert distance to score
    fn distance_to_score(distance: f64) -> f64 {
        // Similar to z-score conversion
        1.0 - (-0.5 * distance).exp()
    }
}

impl Default for RollingMahalanobisDetector {
    fn default() -> Self {
        Self::new(30, 3.0)
    }
}

impl AnomalyDetector for RollingMahalanobisDetector {
    fn detect(&self, data: &[f64]) -> Vec<f64> {
        if data.len() < self.window {
            return vec![0.0; data.len()];
        }

        let mut result = vec![0.0; self.window - 1];

        for i in (self.window - 1)..data.len() {
            let window_data = &data[(i + 1 - self.window)..i];
            let distance = self.rolling_distance(data[i], window_data);
            result.push(Self::distance_to_score(distance));
        }

        result
    }

    fn is_anomaly(&self, score: f64) -> bool {
        score > self.threshold()
    }

    fn threshold(&self) -> f64 {
        Self::distance_to_score(self.threshold_sigma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mahalanobis_fit() {
        let data: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0],
            vec![1.5, 2.5],
            vec![1.2, 2.2],
            vec![1.3, 2.3],
            vec![1.1, 2.1],
        ];

        let detector = MahalanobisDetector::fit(&data, 0.95).unwrap();

        // Normal point should have low distance
        let normal = vec![1.2, 2.2];
        let dist_normal = detector.mahalanobis_distance(&normal);

        // Anomaly should have high distance
        let anomaly = vec![10.0, 20.0];
        let dist_anomaly = detector.mahalanobis_distance(&anomaly);

        assert!(dist_anomaly > dist_normal * 5.0);
    }

    #[test]
    fn test_rolling_mahalanobis() {
        let detector = RollingMahalanobisDetector::new(20, 3.0);
        let mut data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1).sin()).collect();
        data.push(200.0); // Obvious anomaly

        let scores = detector.detect(&data);

        assert!(scores.last().unwrap() > &0.9);
    }
}
