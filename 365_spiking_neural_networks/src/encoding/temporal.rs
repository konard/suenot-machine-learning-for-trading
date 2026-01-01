//! Temporal coding encoder
//!
//! Encodes values in spike timing (time-to-first-spike).

use crate::encoding::{SpikeEncoder, normalize};
use crate::neuron::{Spike, SpikePolarity};

/// Temporal (time-to-first-spike) encoder
///
/// Higher values produce earlier spikes.
#[derive(Debug, Clone)]
pub struct TemporalEncoder {
    /// Maximum time for encoding (ms)
    max_time: f64,
    /// Minimum value for normalization
    min_val: f64,
    /// Maximum value for normalization
    max_val: f64,
    /// Whether to invert (higher value = later spike)
    inverted: bool,
}

impl TemporalEncoder {
    /// Create a new temporal encoder
    pub fn new(max_time: f64, min_val: f64, max_val: f64) -> Self {
        Self {
            max_time,
            min_val,
            max_val,
            inverted: false,
        }
    }

    /// Create an encoder for price changes
    pub fn for_price_change() -> Self {
        Self::new(100.0, -0.05, 0.05)  // ±5% range, 100ms window
    }

    /// Create an encoder for signal strength
    pub fn for_signal_strength() -> Self {
        Self::new(50.0, 0.0, 1.0)  // 50ms window
    }

    /// Invert the encoding (higher value = later spike)
    pub fn inverted(mut self) -> Self {
        self.inverted = true;
        self
    }

    /// Get the spike time for a value
    pub fn spike_time(&self, value: f64) -> f64 {
        let normalized = normalize(value, self.min_val, self.max_val);

        if self.inverted {
            normalized * self.max_time
        } else {
            (1.0 - normalized) * self.max_time
        }
    }

    /// Check if a spike should occur at given time
    pub fn should_spike(&self, value: f64, current_time: f64, dt: f64) -> bool {
        let spike_time = self.spike_time(value);
        current_time <= spike_time && spike_time < current_time + dt
    }
}

impl SpikeEncoder for TemporalEncoder {
    fn encode(&self, value: f64) -> f64 {
        // Return normalized value (0-1)
        // Higher value = earlier spike (lower spike time)
        let normalized = normalize(value, self.min_val, self.max_val);
        if self.inverted {
            normalized
        } else {
            1.0 - normalized
        }
    }

    fn generate_spikes(&self, values: &[f64], time: f64) -> Vec<Option<Spike>> {
        let dt = 1.0; // Assume 1ms timestep

        values.iter().map(|&value| {
            if self.should_spike(value, time, dt) {
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

/// Population temporal encoder
///
/// Uses multiple neurons with different preferred values
/// to encode a single value using time-to-first-spike.
#[derive(Debug, Clone)]
pub struct PopulationTemporalEncoder {
    /// Number of neurons in the population
    num_neurons: usize,
    /// Maximum encoding time
    max_time: f64,
    /// Value range
    min_val: f64,
    max_val: f64,
    /// Width of tuning curves (sigma)
    tuning_width: f64,
}

impl PopulationTemporalEncoder {
    /// Create a new population encoder
    pub fn new(num_neurons: usize, max_time: f64, min_val: f64, max_val: f64) -> Self {
        let range = max_val - min_val;
        Self {
            num_neurons,
            max_time,
            min_val,
            max_val,
            tuning_width: range / num_neurons as f64,
        }
    }

    /// Get the preferred value for each neuron
    pub fn preferred_values(&self) -> Vec<f64> {
        let range = self.max_val - self.min_val;
        (0..self.num_neurons)
            .map(|i| {
                self.min_val + range * (i as f64 + 0.5) / self.num_neurons as f64
            })
            .collect()
    }

    /// Encode a value into spike times for each neuron
    pub fn encode_to_times(&self, value: f64) -> Vec<f64> {
        self.preferred_values()
            .iter()
            .map(|&preferred| {
                let diff = (value - preferred).abs();
                let activation = (-diff.powi(2) / (2.0 * self.tuning_width.powi(2))).exp();
                // Higher activation = earlier spike
                (1.0 - activation) * self.max_time
            })
            .collect()
    }

    /// Generate spikes for a value at given time
    pub fn generate_spikes(&self, value: f64, current_time: f64, dt: f64) -> Vec<bool> {
        self.encode_to_times(value)
            .iter()
            .map(|&spike_time| {
                current_time <= spike_time && spike_time < current_time + dt
            })
            .collect()
    }

    /// Get the number of neurons
    pub fn num_neurons(&self) -> usize {
        self.num_neurons
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_encoder() {
        let encoder = TemporalEncoder::new(100.0, 0.0, 1.0);

        // High value should spike early
        let high_time = encoder.spike_time(1.0);
        let low_time = encoder.spike_time(0.0);

        assert!(high_time < low_time);
        assert!((high_time - 0.0).abs() < 1e-10);
        assert!((low_time - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_temporal_encoder_inverted() {
        let encoder = TemporalEncoder::new(100.0, 0.0, 1.0).inverted();

        // High value should spike late when inverted
        let high_time = encoder.spike_time(1.0);
        let low_time = encoder.spike_time(0.0);

        assert!(high_time > low_time);
    }

    #[test]
    fn test_population_encoder() {
        let encoder = PopulationTemporalEncoder::new(10, 50.0, 0.0, 1.0);

        let times = encoder.encode_to_times(0.5);
        assert_eq!(times.len(), 10);

        // Middle neurons should spike earliest for middle value
        let middle_time = times[4]; // or times[5]
        let edge_time = times[0];

        assert!(middle_time < edge_time);
    }

    #[test]
    fn test_preferred_values() {
        let encoder = PopulationTemporalEncoder::new(5, 50.0, 0.0, 1.0);
        let preferred = encoder.preferred_values();

        assert_eq!(preferred.len(), 5);
        assert!((preferred[0] - 0.1).abs() < 1e-10);
        assert!((preferred[2] - 0.5).abs() < 1e-10);
        assert!((preferred[4] - 0.9).abs() < 1e-10);
    }
}
