//! Anomaly detection module for risk monitoring
//!
//! This module provides multiple anomaly detection algorithms:
//! - Z-Score based detection
//! - Isolation Forest approximation
//! - Mahalanobis distance
//! - Ensemble detector combining multiple methods

mod ensemble;
mod isolation_forest;
mod mahalanobis;
mod zscore;

pub use ensemble::*;
pub use isolation_forest::*;
pub use mahalanobis::*;
pub use zscore::*;

/// Common trait for anomaly detectors
pub trait AnomalyDetector {
    /// Detect anomalies and return scores (higher = more anomalous)
    fn detect(&self, data: &[f64]) -> Vec<f64>;

    /// Check if a single value is anomalous
    fn is_anomaly(&self, score: f64) -> bool;

    /// Get the threshold for anomaly classification
    fn threshold(&self) -> f64;
}

/// Anomaly level classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnomalyLevel {
    /// Normal - no action needed
    Normal,
    /// Elevated - watch closely
    Elevated,
    /// High - consider action
    High,
    /// Extreme - immediate action required
    Extreme,
}

impl AnomalyLevel {
    /// Classify based on score (0-1 normalized)
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s < 0.5 => AnomalyLevel::Normal,
            s if s < 0.7 => AnomalyLevel::Elevated,
            s if s < 0.9 => AnomalyLevel::High,
            _ => AnomalyLevel::Extreme,
        }
    }

    /// Get recommended hedge percentage
    pub fn hedge_percentage(&self) -> f64 {
        match self {
            AnomalyLevel::Normal => 0.0,
            AnomalyLevel::Elevated => 0.02,
            AnomalyLevel::High => 0.05,
            AnomalyLevel::Extreme => 0.15,
        }
    }

    /// Get display color code
    pub fn color(&self) -> &'static str {
        match self {
            AnomalyLevel::Normal => "green",
            AnomalyLevel::Elevated => "yellow",
            AnomalyLevel::High => "orange",
            AnomalyLevel::Extreme => "red",
        }
    }
}

/// Result of anomaly detection with context
#[derive(Debug, Clone)]
pub struct AnomalyResult {
    pub score: f64,
    pub level: AnomalyLevel,
    pub zscore_contribution: f64,
    pub isolation_contribution: f64,
    pub mahalanobis_contribution: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl AnomalyResult {
    /// Create a new result
    pub fn new(
        score: f64,
        zscore: f64,
        isolation: f64,
        mahalanobis: f64,
    ) -> Self {
        Self {
            score,
            level: AnomalyLevel::from_score(score),
            zscore_contribution: zscore,
            isolation_contribution: isolation,
            mahalanobis_contribution: mahalanobis,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Get recommended action based on result
    pub fn recommended_action(&self) -> &'static str {
        match self.level {
            AnomalyLevel::Normal => "Continue normal trading",
            AnomalyLevel::Elevated => "Monitor closely, consider reducing leverage",
            AnomalyLevel::High => "Reduce positions, activate light hedge",
            AnomalyLevel::Extreme => "Exit risky positions, activate full hedge",
        }
    }
}
