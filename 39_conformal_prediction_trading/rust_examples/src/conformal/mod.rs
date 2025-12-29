//! Conformal Prediction algorithms
//!
//! This module provides implementations of various conformal prediction methods:
//!
//! - `split` - Split Conformal Prediction (simplest, most practical)
//! - `adaptive` - Adaptive Conformal Inference for time series
//! - `model` - Simple prediction models for use with conformal prediction

pub mod adaptive;
pub mod model;
pub mod split;

pub use adaptive::AdaptiveConformalPredictor;
pub use model::{LinearModel, Model};
pub use split::SplitConformalPredictor;

/// Prediction interval with bounds
#[derive(Debug, Clone)]
pub struct PredictionInterval {
    /// Point prediction (center of interval)
    pub prediction: f64,
    /// Lower bound of interval
    pub lower: f64,
    /// Upper bound of interval
    pub upper: f64,
    /// Width of interval (upper - lower)
    pub width: f64,
}

impl PredictionInterval {
    /// Create a new prediction interval
    pub fn new(prediction: f64, lower: f64, upper: f64) -> Self {
        Self {
            prediction,
            lower,
            upper,
            width: upper - lower,
        }
    }

    /// Check if a value is covered by this interval
    pub fn covers(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }

    /// Get the midpoint of the interval
    pub fn midpoint(&self) -> f64 {
        (self.lower + self.upper) / 2.0
    }

    /// Check if the entire interval is positive
    pub fn is_positive(&self) -> bool {
        self.lower > 0.0
    }

    /// Check if the entire interval is negative
    pub fn is_negative(&self) -> bool {
        self.upper < 0.0
    }

    /// Check if the interval crosses zero
    pub fn crosses_zero(&self) -> bool {
        self.lower <= 0.0 && self.upper >= 0.0
    }
}
