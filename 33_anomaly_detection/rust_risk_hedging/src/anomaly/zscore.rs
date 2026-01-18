//! Z-Score based anomaly detection
//!
//! Detects anomalies by measuring how many standard deviations
//! a value is from the mean (rolling window)

use super::AnomalyDetector;

/// Z-Score based anomaly detector
#[derive(Debug, Clone)]
pub struct ZScoreDetector {
    /// Rolling window size for calculating mean and std
    window: usize,
    /// Number of standard deviations for anomaly threshold
    threshold_sigma: f64,
}

impl ZScoreDetector {
    /// Create a new Z-Score detector
    ///
    /// # Arguments
    /// * `window` - Rolling window size (e.g., 20 for 20 periods)
    /// * `threshold_sigma` - Number of standard deviations (e.g., 3.0)
    pub fn new(window: usize, threshold_sigma: f64) -> Self {
        Self {
            window: window.max(2),
            threshold_sigma: threshold_sigma.abs(),
        }
    }

    /// Calculate Z-score for a value given window data
    pub fn calculate_zscore(value: f64, window_data: &[f64]) -> f64 {
        if window_data.is_empty() {
            return 0.0;
        }

        let n = window_data.len() as f64;
        let mean = window_data.iter().sum::<f64>() / n;
        let variance = window_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        if std < 1e-10 {
            return 0.0;
        }

        (value - mean) / std
    }

    /// Calculate rolling Z-scores for entire series
    pub fn rolling_zscore(&self, data: &[f64]) -> Vec<f64> {
        if data.len() < self.window {
            return vec![0.0; data.len()];
        }

        let mut result = vec![0.0; self.window - 1];

        for i in (self.window - 1)..data.len() {
            let window_data = &data[(i + 1 - self.window)..i];
            let zscore = Self::calculate_zscore(data[i], window_data);
            result.push(zscore);
        }

        result
    }

    /// Convert Z-score to anomaly score (0-1)
    pub fn zscore_to_anomaly_score(zscore: f64) -> f64 {
        // Use absolute value and sigmoid-like transformation
        let abs_z = zscore.abs();
        // Maps z=0 -> 0, z=3 -> ~0.95, z=5 -> ~0.99
        1.0 - (-0.5 * abs_z).exp()
    }
}

impl Default for ZScoreDetector {
    fn default() -> Self {
        Self::new(20, 3.0)
    }
}

impl AnomalyDetector for ZScoreDetector {
    fn detect(&self, data: &[f64]) -> Vec<f64> {
        self.rolling_zscore(data)
            .into_iter()
            .map(|z| Self::zscore_to_anomaly_score(z))
            .collect()
    }

    fn is_anomaly(&self, score: f64) -> bool {
        score > self.threshold()
    }

    fn threshold(&self) -> f64 {
        Self::zscore_to_anomaly_score(self.threshold_sigma)
    }
}

/// Multi-dimensional Z-Score detector for feature vectors
#[derive(Debug, Clone)]
pub struct MultiZScoreDetector {
    window: usize,
    threshold_sigma: f64,
}

impl MultiZScoreDetector {
    /// Create new multi-dimensional detector
    pub fn new(window: usize, threshold_sigma: f64) -> Self {
        Self {
            window: window.max(2),
            threshold_sigma: threshold_sigma.abs(),
        }
    }

    /// Calculate Z-scores for each feature and combine
    pub fn detect_multi(&self, features: &[Vec<f64>]) -> Vec<f64> {
        if features.is_empty() || features[0].is_empty() {
            return Vec::new();
        }

        let n_features = features[0].len();
        let n_samples = features.len();

        // Calculate Z-score for each feature
        let mut feature_zscores: Vec<Vec<f64>> = Vec::with_capacity(n_features);

        for f_idx in 0..n_features {
            let feature_values: Vec<f64> = features.iter().map(|row| row[f_idx]).collect();
            let detector = ZScoreDetector::new(self.window, self.threshold_sigma);
            feature_zscores.push(detector.rolling_zscore(&feature_values));
        }

        // Combine Z-scores using root mean square
        (0..n_samples)
            .map(|i| {
                let sum_sq: f64 = feature_zscores.iter().map(|fz| fz[i].powi(2)).sum();
                let rms = (sum_sq / n_features as f64).sqrt();
                ZScoreDetector::zscore_to_anomaly_score(rms)
            })
            .collect()
    }
}

impl Default for MultiZScoreDetector {
    fn default() -> Self {
        Self::new(20, 3.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zscore_calculation() {
        let window = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let zscore = ZScoreDetector::calculate_zscore(10.0, &window);
        // Mean = 3, Std = sqrt(2) ≈ 1.41
        // Z = (10-3) / 1.41 ≈ 4.95
        assert!(zscore > 4.0);
    }

    #[test]
    fn test_rolling_zscore() {
        let detector = ZScoreDetector::new(5, 3.0);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100 is anomaly

        let zscores = detector.rolling_zscore(&data);

        // The last Z-score should be very high
        assert!(zscores.last().unwrap().abs() > 3.0);
    }

    #[test]
    fn test_anomaly_detection() {
        let detector = ZScoreDetector::new(5, 2.0);
        let data = vec![1.0, 1.1, 0.9, 1.0, 1.1, 10.0]; // 10 is anomaly

        let scores = detector.detect(&data);

        assert!(detector.is_anomaly(*scores.last().unwrap()));
    }
}
