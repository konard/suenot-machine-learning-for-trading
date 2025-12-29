//! Z-Score based anomaly detection
//!
//! Detects anomalies based on how many standard deviations
//! a value is from the mean.

use super::{AnomalyDetector, AnomalyResult};
use crate::data::{rolling_mean, rolling_std};

/// Z-Score anomaly detector
///
/// Flags values that are more than `threshold` standard deviations
/// from the rolling mean as anomalies.
#[derive(Clone, Debug)]
pub struct ZScoreDetector {
    /// Window size for rolling statistics
    pub window: usize,
    /// Threshold in standard deviations
    pub threshold: f64,
}

impl ZScoreDetector {
    /// Create a new Z-Score detector
    ///
    /// # Arguments
    /// * `window` - Size of the rolling window
    /// * `threshold` - Number of standard deviations for anomaly threshold
    pub fn new(window: usize, threshold: f64) -> Self {
        Self { window, threshold }
    }

    /// Create with default parameters (window=20, threshold=3.0)
    pub fn default_params() -> Self {
        Self::new(20, 3.0)
    }

    /// Compute Z-scores for the data
    pub fn compute_zscores(&self, data: &[f64]) -> Vec<f64> {
        let means = rolling_mean(data, self.window);
        let stds = rolling_std(data, self.window);

        data.iter()
            .zip(means.iter().zip(stds.iter()))
            .map(|(&val, (&mean, &std))| {
                if mean.is_nan() || std.is_nan() || std < 1e-10 {
                    f64::NAN
                } else {
                    (val - mean) / std
                }
            })
            .collect()
    }
}

impl AnomalyDetector for ZScoreDetector {
    fn detect(&self, data: &[f64]) -> AnomalyResult {
        let zscores = self.compute_zscores(data);

        let is_anomaly: Vec<bool> = zscores
            .iter()
            .map(|&z| !z.is_nan() && z.abs() > self.threshold)
            .collect();

        let scores: Vec<f64> = zscores
            .iter()
            .map(|&z| if z.is_nan() { 0.0 } else { z.abs() })
            .collect();

        let normalized_scores: Vec<f64> = scores
            .iter()
            .map(|&s| s / self.threshold)
            .collect();

        AnomalyResult::new(is_anomaly, scores, normalized_scores)
    }

    fn name(&self) -> &str {
        "ZScore"
    }
}

/// Global Z-Score detector (uses entire dataset statistics)
#[derive(Clone, Debug)]
pub struct GlobalZScoreDetector {
    /// Threshold in standard deviations
    pub threshold: f64,
    /// Pre-computed mean (optional, for online use)
    mean: Option<f64>,
    /// Pre-computed std (optional, for online use)
    std: Option<f64>,
}

impl GlobalZScoreDetector {
    /// Create a new global Z-Score detector
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            mean: None,
            std: None,
        }
    }

    /// Fit the detector to training data
    pub fn fit(&mut self, data: &[f64]) {
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        self.mean = Some(mean);
        self.std = Some(std);
    }

    /// Compute Z-score for a single value
    pub fn score_single(&self, value: f64) -> f64 {
        match (self.mean, self.std) {
            (Some(mean), Some(std)) if std > 1e-10 => (value - mean) / std,
            _ => 0.0,
        }
    }

    /// Check if a single value is an anomaly
    pub fn is_anomaly_single(&self, value: f64) -> bool {
        self.score_single(value).abs() > self.threshold
    }
}

impl AnomalyDetector for GlobalZScoreDetector {
    fn detect(&self, data: &[f64]) -> AnomalyResult {
        let n = data.len() as f64;

        // Use pre-fitted statistics or compute from data
        let (mean, std) = match (self.mean, self.std) {
            (Some(m), Some(s)) => (m, s),
            _ => {
                let mean = data.iter().sum::<f64>() / n;
                let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
                (mean, variance.sqrt())
            }
        };

        let zscores: Vec<f64> = data
            .iter()
            .map(|&x| {
                if std > 1e-10 {
                    (x - mean) / std
                } else {
                    0.0
                }
            })
            .collect();

        let is_anomaly: Vec<bool> = zscores
            .iter()
            .map(|&z| z.abs() > self.threshold)
            .collect();

        let scores: Vec<f64> = zscores.iter().map(|&z| z.abs()).collect();

        let normalized_scores: Vec<f64> = scores
            .iter()
            .map(|&s| s / self.threshold)
            .collect();

        AnomalyResult::new(is_anomaly, scores, normalized_scores)
    }

    fn name(&self) -> &str {
        "GlobalZScore"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zscore_detector() {
        let data: Vec<f64> = (0..100)
            .map(|i| if i == 50 { 100.0 } else { (i as f64).sin() })
            .collect();

        let detector = ZScoreDetector::new(20, 3.0);
        let result = detector.detect(&data);

        // Should detect the spike at index 50
        assert!(result.is_anomaly[50]);
        assert!(result.anomaly_count() > 0);
    }

    #[test]
    fn test_global_zscore_detector() {
        let mut data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        data[50] = 1000.0; // Outlier

        let mut detector = GlobalZScoreDetector::new(3.0);
        detector.fit(&data);
        let result = detector.detect(&data);

        assert!(result.is_anomaly[50]);
    }

    #[test]
    fn test_zscore_with_constant_data() {
        let data = vec![5.0; 100];
        let detector = ZScoreDetector::new(20, 3.0);
        let result = detector.detect(&data);

        // No anomalies in constant data
        assert_eq!(result.anomaly_count(), 0);
    }
}
