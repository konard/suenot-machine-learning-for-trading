//! IQR (Interquartile Range) based anomaly detection
//!
//! Detects outliers based on the interquartile range method.
//! Values below Q1 - k*IQR or above Q3 + k*IQR are considered anomalies.

use super::{AnomalyDetector, AnomalyResult};

/// IQR-based anomaly detector
#[derive(Clone, Debug)]
pub struct IQRDetector {
    /// Multiplier for IQR (typically 1.5 for outliers, 3.0 for extreme outliers)
    pub k: f64,
    /// Window size for rolling calculations (0 = use global statistics)
    pub window: usize,
}

impl IQRDetector {
    /// Create a new IQR detector
    ///
    /// # Arguments
    /// * `k` - Multiplier for IQR (1.5 = outliers, 3.0 = extreme outliers)
    /// * `window` - Window size (0 for global)
    pub fn new(k: f64, window: usize) -> Self {
        Self { k, window }
    }

    /// Create with default parameters (k=1.5, global)
    pub fn default_params() -> Self {
        Self::new(1.5, 0)
    }

    /// Create for extreme outliers (k=3.0)
    pub fn extreme() -> Self {
        Self::new(3.0, 0)
    }

    /// Compute Q1, Q3, and IQR for a slice of data
    fn compute_quartiles(data: &[f64]) -> (f64, f64, f64) {
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted.len();
        let q1_idx = n / 4;
        let q3_idx = 3 * n / 4;

        let q1 = if n % 4 == 0 {
            (sorted[q1_idx - 1] + sorted[q1_idx]) / 2.0
        } else {
            sorted[q1_idx]
        };

        let q3 = if 3 * n % 4 == 0 {
            (sorted[q3_idx - 1] + sorted[q3_idx]) / 2.0
        } else {
            sorted[q3_idx]
        };

        let iqr = q3 - q1;
        (q1, q3, iqr)
    }

    /// Compute bounds for anomaly detection
    fn compute_bounds(&self, data: &[f64]) -> (f64, f64) {
        let (q1, q3, iqr) = Self::compute_quartiles(data);
        let lower = q1 - self.k * iqr;
        let upper = q3 + self.k * iqr;
        (lower, upper)
    }
}

impl AnomalyDetector for IQRDetector {
    fn detect(&self, data: &[f64]) -> AnomalyResult {
        if data.is_empty() {
            return AnomalyResult::new(vec![], vec![], vec![]);
        }

        let n = data.len();
        let mut is_anomaly = vec![false; n];
        let mut scores = vec![0.0; n];

        if self.window == 0 || self.window >= n {
            // Global IQR
            let (lower, upper) = self.compute_bounds(data);
            let (_, _, iqr) = Self::compute_quartiles(data);

            for (i, &val) in data.iter().enumerate() {
                if val < lower {
                    is_anomaly[i] = true;
                    scores[i] = if iqr > 0.0 {
                        (lower - val) / iqr
                    } else {
                        1.0
                    };
                } else if val > upper {
                    is_anomaly[i] = true;
                    scores[i] = if iqr > 0.0 {
                        (val - upper) / iqr
                    } else {
                        1.0
                    };
                }
            }
        } else {
            // Rolling IQR
            for i in self.window - 1..n {
                let window_data: &[f64] = &data[i + 1 - self.window..=i];
                let (lower, upper) = self.compute_bounds(window_data);
                let (_, _, iqr) = Self::compute_quartiles(window_data);

                let val = data[i];
                if val < lower {
                    is_anomaly[i] = true;
                    scores[i] = if iqr > 0.0 {
                        (lower - val) / iqr
                    } else {
                        1.0
                    };
                } else if val > upper {
                    is_anomaly[i] = true;
                    scores[i] = if iqr > 0.0 {
                        (val - upper) / iqr
                    } else {
                        1.0
                    };
                }
            }
        }

        // Normalized scores (score / k gives relative position beyond bounds)
        let normalized_scores: Vec<f64> = scores.iter().map(|&s| s / self.k).collect();

        AnomalyResult::new(is_anomaly, scores, normalized_scores)
    }

    fn name(&self) -> &str {
        "IQR"
    }
}

/// Tukey's fences for outlier detection
#[derive(Clone, Debug)]
pub struct TukeyDetector {
    /// Inner fence multiplier (typically 1.5)
    pub inner_k: f64,
    /// Outer fence multiplier (typically 3.0)
    pub outer_k: f64,
}

impl TukeyDetector {
    /// Create a new Tukey detector with default fences
    pub fn new() -> Self {
        Self {
            inner_k: 1.5,
            outer_k: 3.0,
        }
    }

    /// Create with custom fence multipliers
    pub fn with_fences(inner_k: f64, outer_k: f64) -> Self {
        Self { inner_k, outer_k }
    }

    /// Classify an observation
    ///
    /// Returns: 0 = normal, 1 = mild outlier, 2 = extreme outlier
    pub fn classify(&self, value: f64, q1: f64, q3: f64, iqr: f64) -> u8 {
        let inner_lower = q1 - self.inner_k * iqr;
        let inner_upper = q3 + self.inner_k * iqr;
        let outer_lower = q1 - self.outer_k * iqr;
        let outer_upper = q3 + self.outer_k * iqr;

        if value < outer_lower || value > outer_upper {
            2 // Extreme outlier
        } else if value < inner_lower || value > inner_upper {
            1 // Mild outlier
        } else {
            0 // Normal
        }
    }
}

impl Default for TukeyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl AnomalyDetector for TukeyDetector {
    fn detect(&self, data: &[f64]) -> AnomalyResult {
        if data.is_empty() {
            return AnomalyResult::new(vec![], vec![], vec![]);
        }

        let (q1, q3, iqr) = IQRDetector::compute_quartiles(data);

        let classifications: Vec<u8> = data
            .iter()
            .map(|&v| self.classify(v, q1, q3, iqr))
            .collect();

        let is_anomaly: Vec<bool> = classifications.iter().map(|&c| c > 0).collect();

        let scores: Vec<f64> = classifications.iter().map(|&c| c as f64).collect();

        let normalized_scores: Vec<f64> = scores.iter().map(|&s| s / 2.0).collect();

        AnomalyResult::new(is_anomaly, scores, normalized_scores)
    }

    fn name(&self) -> &str {
        "Tukey"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iqr_detector() {
        let mut data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        data[50] = 1000.0; // Extreme outlier

        let detector = IQRDetector::new(1.5, 0);
        let result = detector.detect(&data);

        assert!(result.is_anomaly[50]);
    }

    #[test]
    fn test_rolling_iqr() {
        let mut data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        data[80] = 100.0; // Outlier

        let detector = IQRDetector::new(1.5, 20);
        let result = detector.detect(&data);

        assert!(result.is_anomaly[80]);
    }

    #[test]
    fn test_tukey_classification() {
        let detector = TukeyDetector::new();

        // Q1=25, Q3=75, IQR=50
        // Inner fences: 25-75 = -50, 75+75 = 150
        // Outer fences: 25-150 = -125, 75+150 = 225

        assert_eq!(detector.classify(50.0, 25.0, 75.0, 50.0), 0); // Normal
        assert_eq!(detector.classify(-60.0, 25.0, 75.0, 50.0), 1); // Mild outlier
        assert_eq!(detector.classify(-200.0, 25.0, 75.0, 50.0), 2); // Extreme outlier
    }
}
