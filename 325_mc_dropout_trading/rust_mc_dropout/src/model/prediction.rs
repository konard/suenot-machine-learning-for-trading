//! Prediction result types for MC Dropout

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Result of MC Dropout prediction with uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    /// Mean prediction (expected value)
    pub mean: f64,
    /// Standard deviation (uncertainty)
    pub std: f64,
    /// Variance
    pub variance: f64,
    /// All MC samples
    pub samples: Vec<f64>,
    /// Number of MC samples used
    pub n_samples: usize,
}

impl PredictionResult {
    /// Create prediction result from MC samples
    pub fn from_samples(samples: Vec<f64>) -> Self {
        let n = samples.len() as f64;
        if n == 0.0 {
            return Self {
                mean: 0.0,
                std: 0.0,
                variance: 0.0,
                samples: Vec::new(),
                n_samples: 0,
            };
        }

        let mean: f64 = samples.iter().sum::<f64>() / n;
        let variance: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        Self {
            mean,
            std,
            variance,
            samples: samples.clone(),
            n_samples: samples.len(),
        }
    }

    /// Calculate z-score (confidence measure)
    pub fn z_score(&self) -> f64 {
        if self.std > 0.0 {
            self.mean.abs() / self.std
        } else {
            0.0
        }
    }

    /// Get confidence interval
    pub fn confidence_interval(&self, confidence: f64) -> (f64, f64) {
        // Using normal approximation
        let z = match confidence {
            c if c >= 0.99 => 2.576,
            c if c >= 0.95 => 1.96,
            c if c >= 0.90 => 1.645,
            c if c >= 0.80 => 1.282,
            _ => 1.0,
        };

        let lower = self.mean - z * self.std;
        let upper = self.mean + z * self.std;
        (lower, upper)
    }

    /// Check if prediction is confident (z-score > threshold)
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.z_score() > threshold
    }

    /// Get percentile from samples
    pub fn percentile(&self, p: f64) -> f64 {
        if self.samples.is_empty() {
            return self.mean;
        }

        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
}

/// Result for batch predictions
#[derive(Debug, Clone)]
pub struct BatchPredictionResult {
    /// Individual predictions
    pub predictions: Vec<PredictionResult>,
}

impl BatchPredictionResult {
    /// Create from individual results
    pub fn from_predictions(predictions: Vec<PredictionResult>) -> Self {
        Self { predictions }
    }

    /// Get means as array
    pub fn means(&self) -> Array1<f64> {
        Array1::from_iter(self.predictions.iter().map(|p| p.mean))
    }

    /// Get standard deviations as array
    pub fn stds(&self) -> Array1<f64> {
        Array1::from_iter(self.predictions.iter().map(|p| p.std))
    }

    /// Get z-scores as array
    pub fn z_scores(&self) -> Array1<f64> {
        Array1::from_iter(self.predictions.iter().map(|p| p.z_score()))
    }

    /// Filter predictions by confidence
    pub fn filter_by_confidence(&self, threshold: f64) -> Vec<(usize, &PredictionResult)> {
        self.predictions
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_confident(threshold))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_from_samples() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = PredictionResult::from_samples(samples);

        assert!((result.mean - 3.0).abs() < 1e-10);
        assert!(result.std > 0.0);
        assert_eq!(result.n_samples, 5);
    }

    #[test]
    fn test_confidence_interval() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = PredictionResult::from_samples(samples);

        let (lower, upper) = result.confidence_interval(0.95);
        assert!(lower < result.mean);
        assert!(upper > result.mean);
    }
}
