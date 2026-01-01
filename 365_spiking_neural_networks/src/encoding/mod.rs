//! Spike encoding schemes
//!
//! This module provides various methods to convert continuous values
//! (like prices, volumes) into spike trains.

mod rate;
mod temporal;
mod delta;

pub use rate::RateEncoder;
pub use temporal::TemporalEncoder;
pub use delta::DeltaEncoder;

use crate::neuron::Spike;

/// Common trait for spike encoders
pub trait SpikeEncoder: Send + Sync {
    /// Encode a single value into spike probability/timing
    fn encode(&self, value: f64) -> f64;

    /// Encode a vector of values
    fn encode_batch(&self, values: &[f64]) -> Vec<f64> {
        values.iter().map(|&v| self.encode(v)).collect()
    }

    /// Generate spikes from values for a given timestep
    fn generate_spikes(&self, values: &[f64], time: f64) -> Vec<Option<Spike>>;
}

/// Normalize values to [0, 1] range
pub fn normalize(value: f64, min: f64, max: f64) -> f64 {
    if max <= min {
        return 0.5;
    }
    ((value - min) / (max - min)).clamp(0.0, 1.0)
}

/// Normalize using z-score
pub fn z_score_normalize(value: f64, mean: f64, std: f64) -> f64 {
    if std <= 0.0 {
        return 0.0;
    }
    (value - mean) / std
}

/// Convert z-score to probability using sigmoid
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        assert!((normalize(50.0, 0.0, 100.0) - 0.5).abs() < 1e-10);
        assert!((normalize(0.0, 0.0, 100.0) - 0.0).abs() < 1e-10);
        assert!((normalize(100.0, 0.0, 100.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }
}
