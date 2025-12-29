//! Modified Z-Score (MAD) based anomaly detection
//!
//! Uses Median Absolute Deviation for robust anomaly detection.
//! More resistant to outliers than standard Z-score.

use super::{AnomalyDetector, AnomalyResult};
use crate::data::{rolling_mad, rolling_median};

/// Modified Z-Score detector using Median Absolute Deviation
///
/// The modified Z-score is computed as:
/// M_i = 0.6745 * (x_i - median) / MAD
///
/// where 0.6745 is a scaling factor for consistency with standard deviation.
#[derive(Clone, Debug)]
pub struct MADDetector {
    /// Window size for rolling calculations
    pub window: usize,
    /// Threshold for anomaly detection
    pub threshold: f64,
    /// Scaling factor for consistency with normal distribution
    pub scale: f64,
}

impl MADDetector {
    /// Create a new MAD detector
    ///
    /// # Arguments
    /// * `window` - Size of the rolling window
    /// * `threshold` - Threshold for modified Z-score (typical: 3.5)
    pub fn new(window: usize, threshold: f64) -> Self {
        Self {
            window,
            threshold,
            scale: 0.6745, // Consistency factor for normal distribution
        }
    }

    /// Create with default parameters (window=20, threshold=3.5)
    pub fn default_params() -> Self {
        Self::new(20, 3.5)
    }

    /// Compute modified Z-scores
    pub fn compute_modified_zscores(&self, data: &[f64]) -> Vec<f64> {
        let medians = rolling_median(data, self.window);
        let mads = rolling_mad(data, self.window);

        data.iter()
            .zip(medians.iter().zip(mads.iter()))
            .map(|(&val, (&median, &mad))| {
                if median.is_nan() || mad.is_nan() || mad < 1e-10 {
                    f64::NAN
                } else {
                    self.scale * (val - median) / mad
                }
            })
            .collect()
    }
}

impl AnomalyDetector for MADDetector {
    fn detect(&self, data: &[f64]) -> AnomalyResult {
        let modified_zscores = self.compute_modified_zscores(data);

        let is_anomaly: Vec<bool> = modified_zscores
            .iter()
            .map(|&z| !z.is_nan() && z.abs() > self.threshold)
            .collect();

        let scores: Vec<f64> = modified_zscores
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
        "MAD"
    }
}

/// Global MAD detector (uses entire dataset)
#[derive(Clone, Debug)]
pub struct GlobalMADDetector {
    /// Threshold for anomaly detection
    pub threshold: f64,
    /// Scaling factor
    pub scale: f64,
    /// Pre-computed median
    median: Option<f64>,
    /// Pre-computed MAD
    mad: Option<f64>,
}

impl GlobalMADDetector {
    /// Create a new global MAD detector
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            scale: 0.6745,
            median: None,
            mad: None,
        }
    }

    /// Fit the detector to training data
    pub fn fit(&mut self, data: &[f64]) {
        // Compute median
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        // Compute MAD
        let mut deviations: Vec<f64> = data.iter().map(|x| (x - median).abs()).collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = if n % 2 == 0 {
            (deviations[n / 2 - 1] + deviations[n / 2]) / 2.0
        } else {
            deviations[n / 2]
        };

        self.median = Some(median);
        self.mad = Some(mad);
    }

    /// Compute modified Z-score for a single value
    pub fn score_single(&self, value: f64) -> f64 {
        match (self.median, self.mad) {
            (Some(median), Some(mad)) if mad > 1e-10 => {
                self.scale * (value - median) / mad
            }
            _ => 0.0,
        }
    }

    /// Check if a single value is an anomaly
    pub fn is_anomaly_single(&self, value: f64) -> bool {
        self.score_single(value).abs() > self.threshold
    }
}

impl AnomalyDetector for GlobalMADDetector {
    fn detect(&self, data: &[f64]) -> AnomalyResult {
        // Use pre-fitted statistics or compute from data
        let (median, mad) = match (self.median, self.mad) {
            (Some(m), Some(d)) => (m, d),
            _ => {
                let mut sorted = data.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = sorted.len();
                let med = if n % 2 == 0 {
                    (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                } else {
                    sorted[n / 2]
                };

                let mut devs: Vec<f64> = data.iter().map(|x| (x - med).abs()).collect();
                devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let m = if n % 2 == 0 {
                    (devs[n / 2 - 1] + devs[n / 2]) / 2.0
                } else {
                    devs[n / 2]
                };

                (med, m)
            }
        };

        let modified_zscores: Vec<f64> = data
            .iter()
            .map(|&x| {
                if mad > 1e-10 {
                    self.scale * (x - median) / mad
                } else {
                    0.0
                }
            })
            .collect();

        let is_anomaly: Vec<bool> = modified_zscores
            .iter()
            .map(|&z| z.abs() > self.threshold)
            .collect();

        let scores: Vec<f64> = modified_zscores.iter().map(|&z| z.abs()).collect();

        let normalized_scores: Vec<f64> = scores
            .iter()
            .map(|&s| s / self.threshold)
            .collect();

        AnomalyResult::new(is_anomaly, scores, normalized_scores)
    }

    fn name(&self) -> &str {
        "GlobalMAD"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mad_detector() {
        let mut data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        data[50] = 100.0; // Outlier

        let detector = MADDetector::new(20, 3.5);
        let result = detector.detect(&data);

        // Should detect the outlier
        assert!(result.is_anomaly[50]);
    }

    #[test]
    fn test_global_mad_detector() {
        let mut data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        data[50] = 1000.0; // Outlier

        let mut detector = GlobalMADDetector::new(3.5);
        detector.fit(&data);
        let result = detector.detect(&data);

        assert!(result.is_anomaly[50]);
    }

    #[test]
    fn test_mad_robust_to_outliers() {
        // MAD should be robust to multiple outliers
        let mut data: Vec<f64> = vec![1.0; 100];
        data[50] = 100.0;
        data[51] = 100.0;
        data[52] = 100.0;

        let mut detector = GlobalMADDetector::new(3.5);
        detector.fit(&data);

        // The median and MAD should still be based on the majority (1.0s)
        assert!(detector.median.unwrap() < 10.0);
    }
}
