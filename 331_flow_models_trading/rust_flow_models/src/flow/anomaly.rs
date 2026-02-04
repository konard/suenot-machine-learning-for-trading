//! Anomaly detection using flow model likelihood

use crate::flow::model::NormalizingFlow;
use ndarray::{Array1, Array2};

/// Anomaly detector using flow model density estimation
#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    /// Threshold percentile for anomaly detection
    pub threshold_percentile: f64,
    /// Computed threshold value
    pub threshold: Option<f64>,
    /// Reference log probabilities for calibration
    reference_log_probs: Option<Array1<f64>>,
}

impl AnomalyDetector {
    /// Create new anomaly detector
    ///
    /// # Arguments
    /// * `threshold_percentile` - Percentile for anomaly threshold (e.g., 5.0 means bottom 5% are anomalies)
    pub fn new(threshold_percentile: f64) -> Self {
        Self {
            threshold_percentile,
            threshold: None,
            reference_log_probs: None,
        }
    }

    /// Fit threshold on reference (normal) data
    pub fn fit(&mut self, model: &mut NormalizingFlow, reference_data: &Array2<f64>) {
        // Compute log probabilities for reference data
        let log_probs = model.log_prob(reference_data);

        // Sort and find percentile
        let mut sorted_probs: Vec<f64> = log_probs.to_vec();
        sorted_probs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((self.threshold_percentile / 100.0) * sorted_probs.len() as f64) as usize;
        let idx = idx.min(sorted_probs.len().saturating_sub(1));

        self.threshold = Some(sorted_probs[idx]);
        self.reference_log_probs = Some(log_probs);

        log::info!(
            "Anomaly threshold set to {:.4} at {}th percentile",
            self.threshold.unwrap(),
            self.threshold_percentile
        );
    }

    /// Detect anomalies in new data
    ///
    /// Returns (is_anomaly, log_probs) tuple
    pub fn detect(
        &self,
        model: &mut NormalizingFlow,
        data: &Array2<f64>,
    ) -> (Vec<bool>, Array1<f64>) {
        let log_probs = model.log_prob(data);

        let threshold = self.threshold.unwrap_or(f64::NEG_INFINITY);
        let is_anomaly: Vec<bool> = log_probs.iter().map(|&lp| lp < threshold).collect();

        (is_anomaly, log_probs)
    }

    /// Check if a single sample is anomalous
    pub fn is_anomaly(&self, model: &mut NormalizingFlow, sample: &Array1<f64>) -> (bool, f64) {
        let sample_2d = sample.clone().insert_axis(ndarray::Axis(0));
        let log_probs = model.log_prob(&sample_2d);
        let log_prob = log_probs[0];

        let threshold = self.threshold.unwrap_or(f64::NEG_INFINITY);
        (log_prob < threshold, log_prob)
    }

    /// Get anomaly score (negative log probability - higher means more anomalous)
    pub fn anomaly_score(&self, model: &mut NormalizingFlow, data: &Array2<f64>) -> Array1<f64> {
        let log_probs = model.log_prob(data);
        log_probs.mapv(|lp| -lp)
    }

    /// Get threshold value
    pub fn get_threshold(&self) -> Option<f64> {
        self.threshold
    }

    /// Get statistics about the reference distribution
    pub fn reference_stats(&self) -> Option<(f64, f64, f64, f64)> {
        self.reference_log_probs.as_ref().map(|lps| {
            let mean = lps.mean().unwrap_or(0.0);
            let std = lps.std(0.0);
            let min = lps.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = lps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (mean, std, min, max)
        })
    }
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::new(5.0) // Default to 5th percentile
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::config::FlowConfig;
    use ndarray::Array2;

    #[test]
    fn test_anomaly_detector() {
        let config = FlowConfig::default().with_input_dim(4).with_num_layers(2);
        let mut model = NormalizingFlow::new(config);

        // Create normal data
        let normal_data = Array2::from_shape_fn((100, 4), |(i, j)| {
            ((i + j) as f64) * 0.01
        });

        // Fit detector
        let mut detector = AnomalyDetector::new(10.0);
        detector.fit(&mut model, &normal_data);

        assert!(detector.threshold.is_some());

        // Test detection
        let (is_anomaly, _) = detector.detect(&mut model, &normal_data);
        assert_eq!(is_anomaly.len(), 100);

        // Approximately 10% should be flagged as anomalies
        let anomaly_rate = is_anomaly.iter().filter(|&&a| a).count() as f64 / 100.0;
        assert!(anomaly_rate <= 0.15); // Allow some margin
    }
}
