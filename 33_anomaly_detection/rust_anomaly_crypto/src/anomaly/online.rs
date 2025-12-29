//! Online (streaming) anomaly detection
//!
//! Detectors that work on streaming data without needing to refit
//! on the entire dataset.

use super::AnomalyResult;
use std::collections::VecDeque;

/// Online anomaly detector using rolling statistics
#[derive(Clone, Debug)]
pub struct OnlineDetector {
    /// Window size for rolling calculations
    window: usize,
    /// Threshold for anomaly detection
    threshold: f64,
    /// Buffer for recent values
    buffer: VecDeque<f64>,
    /// Running sum for efficiency
    sum: f64,
    /// Running sum of squares for efficiency
    sum_sq: f64,
}

impl OnlineDetector {
    /// Create a new online detector
    ///
    /// # Arguments
    /// * `window` - Size of the rolling window
    /// * `threshold` - Z-score threshold for anomalies
    pub fn new(window: usize, threshold: f64) -> Self {
        Self {
            window,
            threshold,
            buffer: VecDeque::with_capacity(window),
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Create with default parameters (window=100, threshold=3.0)
    pub fn default_params() -> Self {
        Self::new(100, 3.0)
    }

    /// Clear the detector state
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.sum = 0.0;
        self.sum_sq = 0.0;
    }

    /// Get current buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Check if detector is warmed up (has enough data)
    pub fn is_warmed_up(&self) -> bool {
        self.buffer.len() >= self.window / 2
    }

    /// Get current mean
    pub fn mean(&self) -> Option<f64> {
        if self.buffer.is_empty() {
            None
        } else {
            Some(self.sum / self.buffer.len() as f64)
        }
    }

    /// Get current standard deviation
    pub fn std(&self) -> Option<f64> {
        if self.buffer.len() < 2 {
            None
        } else {
            let n = self.buffer.len() as f64;
            let variance = (self.sum_sq / n) - (self.sum / n).powi(2);
            Some(variance.max(0.0).sqrt())
        }
    }

    /// Update with a new value and return anomaly information
    ///
    /// Returns: (anomaly_score, is_anomaly, z_score)
    pub fn update(&mut self, value: f64) -> (f64, bool, f64) {
        // Compute z-score before updating buffer
        let (z_score, is_anomaly) = if self.is_warmed_up() {
            let mean = self.mean().unwrap_or(0.0);
            let std = self.std().unwrap_or(1.0);

            let z = if std > 1e-10 {
                (value - mean) / std
            } else {
                0.0
            };

            (z, z.abs() > self.threshold)
        } else {
            (0.0, false)
        };

        // Update buffer
        if self.buffer.len() >= self.window {
            if let Some(old_value) = self.buffer.pop_front() {
                self.sum -= old_value;
                self.sum_sq -= old_value * old_value;
            }
        }

        self.buffer.push_back(value);
        self.sum += value;
        self.sum_sq += value * value;

        let anomaly_score = z_score.abs() / self.threshold;
        (anomaly_score, is_anomaly, z_score)
    }

    /// Get adaptive threshold based on recent volatility
    pub fn adaptive_threshold(&self, base_threshold: f64) -> f64 {
        if self.buffer.len() < 20 {
            return base_threshold;
        }

        // Compute volatility of recent window
        let recent: Vec<f64> = self.buffer.iter().rev().take(20).cloned().collect();
        let recent_mean: f64 = recent.iter().sum::<f64>() / 20.0;
        let recent_var: f64 = recent.iter().map(|x| (x - recent_mean).powi(2)).sum::<f64>() / 20.0;
        let recent_std = recent_var.sqrt();

        // Compare to historical volatility
        let historical_std = self.std().unwrap_or(1.0);

        if historical_std > 1e-10 {
            let vol_ratio = recent_std / historical_std;
            // Increase threshold in high volatility
            base_threshold * vol_ratio.max(1.0)
        } else {
            base_threshold
        }
    }
}

/// Online detector with exponential moving average
#[derive(Clone, Debug)]
pub struct EMAOnlineDetector {
    /// Smoothing factor for EMA (0 < alpha < 1)
    alpha: f64,
    /// Threshold for anomaly detection
    threshold: f64,
    /// Current EMA of values
    ema: Option<f64>,
    /// Current EMA of squared deviations (for variance)
    ema_var: Option<f64>,
    /// Number of observations
    n: usize,
    /// Minimum observations before detection
    min_observations: usize,
}

impl EMAOnlineDetector {
    /// Create a new EMA-based online detector
    ///
    /// # Arguments
    /// * `alpha` - Smoothing factor (higher = more weight on recent values)
    /// * `threshold` - Threshold for anomaly detection
    pub fn new(alpha: f64, threshold: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.001, 0.999),
            threshold,
            ema: None,
            ema_var: None,
            n: 0,
            min_observations: 10,
        }
    }

    /// Create from span (like pandas EMA)
    ///
    /// alpha = 2 / (span + 1)
    pub fn from_span(span: usize, threshold: f64) -> Self {
        let alpha = 2.0 / (span as f64 + 1.0);
        Self::new(alpha, threshold)
    }

    /// Reset the detector state
    pub fn reset(&mut self) {
        self.ema = None;
        self.ema_var = None;
        self.n = 0;
    }

    /// Check if detector is warmed up
    pub fn is_warmed_up(&self) -> bool {
        self.n >= self.min_observations
    }

    /// Update with a new value and return anomaly information
    pub fn update(&mut self, value: f64) -> (f64, bool, f64) {
        self.n += 1;

        let (z_score, is_anomaly) = match (self.ema, self.ema_var) {
            (Some(ema), Some(ema_var)) if self.is_warmed_up() => {
                let std = ema_var.sqrt();
                let z = if std > 1e-10 {
                    (value - ema) / std
                } else {
                    0.0
                };
                (z, z.abs() > self.threshold)
            }
            _ => (0.0, false),
        };

        // Update EMA
        match self.ema {
            None => {
                self.ema = Some(value);
                self.ema_var = Some(0.0);
            }
            Some(ema) => {
                let deviation = value - ema;
                let new_ema = ema + self.alpha * deviation;

                // Update variance EMA
                let var = self.ema_var.unwrap_or(0.0);
                let new_var = (1.0 - self.alpha) * (var + self.alpha * deviation * deviation);

                self.ema = Some(new_ema);
                self.ema_var = Some(new_var);
            }
        }

        let anomaly_score = z_score.abs() / self.threshold;
        (anomaly_score, is_anomaly, z_score)
    }

    /// Get current EMA
    pub fn mean(&self) -> Option<f64> {
        self.ema
    }

    /// Get current standard deviation
    pub fn std(&self) -> Option<f64> {
        self.ema_var.map(|v| v.sqrt())
    }
}

/// Multi-feature online detector
#[derive(Clone)]
pub struct MultivariateOnlineDetector {
    /// Individual detectors for each feature
    detectors: Vec<OnlineDetector>,
    /// Feature weights for combining scores
    weights: Vec<f64>,
    /// Combined threshold
    threshold: f64,
}

impl MultivariateOnlineDetector {
    /// Create a new multivariate online detector
    ///
    /// # Arguments
    /// * `n_features` - Number of features
    /// * `window` - Window size for each detector
    /// * `threshold` - Threshold for anomaly detection
    pub fn new(n_features: usize, window: usize, threshold: f64) -> Self {
        let detectors = (0..n_features)
            .map(|_| OnlineDetector::new(window, threshold))
            .collect();

        let weights = vec![1.0 / n_features as f64; n_features];

        Self {
            detectors,
            weights,
            threshold,
        }
    }

    /// Set weights for features
    pub fn set_weights(&mut self, weights: Vec<f64>) {
        assert_eq!(weights.len(), self.detectors.len());
        let sum: f64 = weights.iter().sum();
        self.weights = weights.into_iter().map(|w| w / sum).collect();
    }

    /// Reset all detectors
    pub fn reset(&mut self) {
        for detector in &mut self.detectors {
            detector.reset();
        }
    }

    /// Check if all detectors are warmed up
    pub fn is_warmed_up(&self) -> bool {
        self.detectors.iter().all(|d| d.is_warmed_up())
    }

    /// Update with new feature values
    ///
    /// Returns: (combined_score, is_anomaly, individual_scores)
    pub fn update(&mut self, values: &[f64]) -> (f64, bool, Vec<f64>) {
        assert_eq!(values.len(), self.detectors.len());

        let individual_scores: Vec<f64> = self
            .detectors
            .iter_mut()
            .zip(values.iter())
            .map(|(detector, &value)| detector.update(value).0)
            .collect();

        let combined_score: f64 = individual_scores
            .iter()
            .zip(self.weights.iter())
            .map(|(&score, &weight)| score * weight)
            .sum();

        let is_anomaly = combined_score > 1.0; // Score > 1 means above threshold

        (combined_score, is_anomaly, individual_scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_online_detector() {
        let mut detector = OnlineDetector::new(20, 3.0);

        // Feed normal data
        for i in 0..50 {
            let value = (i as f64 * 0.1).sin();
            let (_, is_anomaly, _) = detector.update(value);
            if i > 20 {
                assert!(!is_anomaly, "Normal value flagged as anomaly");
            }
        }

        // Feed anomaly
        let (score, is_anomaly, _) = detector.update(100.0);
        assert!(is_anomaly, "Anomaly not detected");
        assert!(score > 1.0);
    }

    #[test]
    fn test_ema_online_detector() {
        let mut detector = EMAOnlineDetector::from_span(20, 3.0);

        // Warmup with normal data
        for i in 0..30 {
            detector.update(i as f64);
        }

        // Mean should be close to recent values
        let mean = detector.mean().unwrap();
        assert!(mean > 10.0 && mean < 30.0);

        // Test anomaly detection
        let (_, is_anomaly, _) = detector.update(1000.0);
        assert!(is_anomaly);
    }

    #[test]
    fn test_multivariate_online() {
        let mut detector = MultivariateOnlineDetector::new(3, 20, 3.0);

        // Warmup
        for i in 0..30 {
            let values = vec![i as f64, (i as f64).sin(), (i as f64).cos()];
            detector.update(&values);
        }

        assert!(detector.is_warmed_up());

        // Test with anomaly in one feature
        let (score, is_anomaly, _) = detector.update(&[100.0, 0.0, 0.0]);
        assert!(score > 0.0);
    }
}
