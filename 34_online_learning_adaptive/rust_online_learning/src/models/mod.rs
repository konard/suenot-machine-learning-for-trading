//! Online Learning Models Module
//!
//! This module provides implementations of online learning algorithms
//! for trading strategies.

mod adaptive_weights;
mod online_linear;

pub use adaptive_weights::AdaptiveMomentumWeights;
pub use online_linear::OnlineLinearRegression;

/// Trait for online learning models
pub trait OnlineModel {
    /// Predict output for input features
    fn predict(&self, x: &[f64]) -> f64;

    /// Learn from a single observation
    fn learn(&mut self, x: &[f64], y: f64);

    /// Get current model parameters
    fn get_params(&self) -> Vec<f64>;

    /// Reset model to initial state
    fn reset(&mut self);
}
