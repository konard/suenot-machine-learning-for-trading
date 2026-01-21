//! Task-specific prediction heads
//!
//! This module provides specialized prediction heads for different trading tasks:
//!
//! - Direction prediction (up/down/sideways)
//! - Volatility estimation
//! - Regime classification
//! - Return prediction

mod direction;
mod volatility;
mod regime;
mod returns;

pub use direction::{Direction, DirectionHead, DirectionConfig, DirectionPrediction};
pub use volatility::{VolatilityLevel, VolatilityHead, VolatilityConfig, VolatilityPrediction};
pub use regime::{RegimeHead, RegimeConfig, RegimePrediction, MarketRegime};
pub use returns::{ReturnsHead, ReturnsConfig, ReturnsPrediction};

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Task type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskType {
    /// Direction prediction (classification)
    Direction,
    /// Volatility estimation (regression)
    Volatility,
    /// Market regime classification
    Regime,
    /// Return prediction (regression)
    Returns,
}

impl std::fmt::Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskType::Direction => write!(f, "Direction"),
            TaskType::Volatility => write!(f, "Volatility"),
            TaskType::Regime => write!(f, "Regime"),
            TaskType::Returns => write!(f, "Returns"),
        }
    }
}

/// Trait for task-specific heads
pub trait TaskHead: Send + Sync {
    /// Get the task type
    fn task_type(&self) -> TaskType;

    /// Forward pass producing raw output
    fn forward(&self, embedding: &Array1<f64>) -> Array1<f64>;

    /// Batch forward pass
    fn forward_batch(&self, embeddings: &Array2<f64>) -> Array2<f64>;

    /// Get task head parameters
    fn parameters(&self) -> Vec<Array2<f64>>;

    /// Update parameters with gradients
    fn update_parameters(&mut self, gradients: &[Array2<f64>], learning_rate: f64);

    /// Compute loss for this task
    fn compute_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64;
}

/// Multi-task prediction result
#[derive(Debug, Clone)]
pub struct MultiTaskPrediction {
    /// Direction prediction
    pub direction: Option<DirectionPrediction>,
    /// Volatility prediction
    pub volatility: Option<VolatilityPrediction>,
    /// Regime prediction
    pub regime: Option<RegimePrediction>,
    /// Returns prediction
    pub returns: Option<ReturnsPrediction>,
}

impl MultiTaskPrediction {
    /// Create empty prediction
    pub fn new() -> Self {
        Self {
            direction: None,
            volatility: None,
            regime: None,
            returns: None,
        }
    }

    /// Check if all tasks are predicted
    pub fn is_complete(&self) -> bool {
        self.direction.is_some()
            && self.volatility.is_some()
            && self.regime.is_some()
            && self.returns.is_some()
    }

    /// Get average confidence across all predictions
    pub fn average_confidence(&self) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;

        if let Some(ref d) = self.direction {
            sum += d.confidence;
            count += 1;
        }
        if let Some(ref v) = self.volatility {
            sum += v.confidence;
            count += 1;
        }
        if let Some(ref r) = self.regime {
            sum += r.confidence;
            count += 1;
        }
        if let Some(ref ret) = self.returns {
            sum += ret.confidence;
            count += 1;
        }

        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }
}

impl Default for MultiTaskPrediction {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_type_display() {
        assert_eq!(format!("{}", TaskType::Direction), "Direction");
        assert_eq!(format!("{}", TaskType::Volatility), "Volatility");
    }

    #[test]
    fn test_multi_task_prediction() {
        let pred = MultiTaskPrediction::new();
        assert!(!pred.is_complete());
        assert_eq!(pred.average_confidence(), 0.0);
    }
}
