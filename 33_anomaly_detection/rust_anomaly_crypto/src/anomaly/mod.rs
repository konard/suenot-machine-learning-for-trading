//! Anomaly detection algorithms
//!
//! This module provides various anomaly detection methods:
//! - Statistical: Z-score, Modified Z-score (MAD), IQR
//! - Machine Learning: Isolation Forest
//! - Ensemble: Combining multiple detectors

mod zscore;
mod mad;
mod iqr;
mod isolation_forest;
mod ensemble;
mod online;

pub use zscore::*;
pub use mad::*;
pub use iqr::*;
pub use isolation_forest::*;
pub use ensemble::*;
pub use online::*;

/// Result of anomaly detection
#[derive(Debug, Clone)]
pub struct AnomalyResult {
    /// Binary flags indicating anomalies
    pub is_anomaly: Vec<bool>,
    /// Continuous anomaly scores (higher = more anomalous)
    pub scores: Vec<f64>,
    /// Z-scores or similar normalized values
    pub normalized_scores: Vec<f64>,
}

impl AnomalyResult {
    /// Create a new anomaly result
    pub fn new(is_anomaly: Vec<bool>, scores: Vec<f64>, normalized_scores: Vec<f64>) -> Self {
        Self {
            is_anomaly,
            scores,
            normalized_scores,
        }
    }

    /// Get indices of anomalies
    pub fn anomaly_indices(&self) -> Vec<usize> {
        self.is_anomaly
            .iter()
            .enumerate()
            .filter_map(|(i, &is_anom)| if is_anom { Some(i) } else { None })
            .collect()
    }

    /// Get the number of detected anomalies
    pub fn anomaly_count(&self) -> usize {
        self.is_anomaly.iter().filter(|&&x| x).count()
    }

    /// Get the anomaly rate
    pub fn anomaly_rate(&self) -> f64 {
        if self.is_anomaly.is_empty() {
            0.0
        } else {
            self.anomaly_count() as f64 / self.is_anomaly.len() as f64
        }
    }

    /// Get the maximum anomaly score
    pub fn max_score(&self) -> f64 {
        self.scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get the mean anomaly score
    pub fn mean_score(&self) -> f64 {
        if self.scores.is_empty() {
            0.0
        } else {
            self.scores.iter().sum::<f64>() / self.scores.len() as f64
        }
    }
}

/// Trait for anomaly detectors
pub trait AnomalyDetector {
    /// Detect anomalies in the given data
    fn detect(&self, data: &[f64]) -> AnomalyResult;

    /// Get the name of the detector
    fn name(&self) -> &str;
}

/// Trait for multivariate anomaly detectors
pub trait MultivariateDetector {
    /// Fit the detector to training data
    fn fit(&mut self, data: &ndarray::Array2<f64>);

    /// Detect anomalies in the given data
    fn detect(&self, data: &ndarray::Array2<f64>) -> AnomalyResult;

    /// Get the name of the detector
    fn name(&self) -> &str;
}
