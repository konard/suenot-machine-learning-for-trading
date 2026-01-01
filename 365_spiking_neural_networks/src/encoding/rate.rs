//! Rate coding encoder
//!
//! Converts values to spike rates (firing frequency).

use crate::encoding::{SpikeEncoder, normalize};
use crate::neuron::{Spike, SpikePolarity};
use rand::Rng;

/// Rate-based spike encoder
///
/// Higher values produce higher spike rates (Poisson-like).
#[derive(Debug, Clone)]
pub struct RateEncoder {
    /// Maximum firing rate (Hz)
    max_rate: f64,
    /// Minimum value for normalization
    min_val: f64,
    /// Maximum value for normalization
    max_val: f64,
    /// Time window for rate calculation (ms)
    time_window: f64,
}

impl RateEncoder {
    /// Create a new rate encoder
    pub fn new(max_rate: f64, min_val: f64, max_val: f64) -> Self {
        Self {
            max_rate,
            min_val,
            max_val,
            time_window: 1000.0, // 1 second default
        }
    }

    /// Create an encoder for price returns (typically small values)
    pub fn for_returns() -> Self {
        Self::new(100.0, -0.1, 0.1)  // ±10% range
    }

    /// Create an encoder for normalized volumes
    pub fn for_volume() -> Self {
        Self::new(200.0, 0.0, 1.0)
    }

    /// Create an encoder for order book imbalance
    pub fn for_imbalance() -> Self {
        Self::new(100.0, -1.0, 1.0)
    }

    /// Set time window
    pub fn with_time_window(mut self, window: f64) -> Self {
        self.time_window = window;
        self
    }

    /// Get expected number of spikes in time window
    pub fn expected_spikes(&self, value: f64) -> f64 {
        let rate = self.encode(value);
        rate * self.time_window / 1000.0
    }

    /// Generate spikes for a time window using Poisson process
    pub fn generate_spike_train(&self, value: f64, duration: f64, dt: f64) -> Vec<f64> {
        let rate = self.encode(value);
        let lambda = rate * dt / 1000.0; // Probability per timestep

        let mut rng = rand::thread_rng();
        let mut spikes = Vec::new();
        let mut t = 0.0;

        while t < duration {
            if rng.gen::<f64>() < lambda {
                spikes.push(t);
            }
            t += dt;
        }

        spikes
    }
}

impl SpikeEncoder for RateEncoder {
    fn encode(&self, value: f64) -> f64 {
        let normalized = normalize(value, self.min_val, self.max_val);
        normalized * self.max_rate
    }

    fn generate_spikes(&self, values: &[f64], time: f64) -> Vec<Option<Spike>> {
        let mut rng = rand::thread_rng();

        values.iter().map(|&value| {
            let rate = self.encode(value);
            let prob = rate / 1000.0; // Convert to probability per ms

            if rng.gen::<f64>() < prob {
                Some(Spike {
                    time,
                    polarity: if value >= (self.min_val + self.max_val) / 2.0 {
                        SpikePolarity::Positive
                    } else {
                        SpikePolarity::Negative
                    },
                    source: None,
                })
            } else {
                None
            }
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_encoder() {
        let encoder = RateEncoder::new(100.0, 0.0, 1.0);

        assert!((encoder.encode(0.0) - 0.0).abs() < 1e-10);
        assert!((encoder.encode(1.0) - 100.0).abs() < 1e-10);
        assert!((encoder.encode(0.5) - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_spike_train_generation() {
        let encoder = RateEncoder::new(100.0, 0.0, 1.0);
        let spikes = encoder.generate_spike_train(0.5, 1000.0, 1.0);

        // With 50 Hz rate over 1 second, expect ~50 spikes
        // Allow for statistical variation
        assert!(spikes.len() > 20 && spikes.len() < 100,
            "Expected ~50 spikes, got {}", spikes.len());
    }

    #[test]
    fn test_for_returns() {
        let encoder = RateEncoder::for_returns();

        // Zero return should be middle rate
        let zero_rate = encoder.encode(0.0);
        assert!((zero_rate - 50.0).abs() < 1e-10);

        // Positive return should be higher
        let pos_rate = encoder.encode(0.05);
        assert!(pos_rate > zero_rate);
    }
}
