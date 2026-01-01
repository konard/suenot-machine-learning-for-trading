//! Delta coding encoder
//!
//! Generates spikes on value changes, similar to event-based vision sensors.
//! Perfect for encoding price changes in trading.

use crate::encoding::SpikeEncoder;
use crate::neuron::{Spike, SpikePolarity};

/// Delta-based spike encoder
///
/// Generates spikes when values change beyond a threshold.
#[derive(Debug, Clone)]
pub struct DeltaEncoder {
    /// Threshold for generating a spike
    threshold: f64,
    /// Previous values for each channel
    previous_values: Vec<f64>,
    /// Whether to use adaptive thresholds
    adaptive: bool,
    /// Adaptation rate
    adaptation_rate: f64,
    /// Per-channel thresholds (for adaptive mode)
    thresholds: Vec<f64>,
}

impl DeltaEncoder {
    /// Create a new delta encoder with fixed threshold
    pub fn new(threshold: f64, num_channels: usize) -> Self {
        Self {
            threshold,
            previous_values: vec![0.0; num_channels],
            adaptive: false,
            adaptation_rate: 0.01,
            thresholds: vec![threshold; num_channels],
        }
    }

    /// Create a delta encoder for price data
    pub fn for_prices(num_channels: usize) -> Self {
        // 0.1% price change threshold
        Self::new(0.001, num_channels)
    }

    /// Create a delta encoder for tick data
    pub fn for_ticks(num_channels: usize) -> Self {
        // Very small threshold for tick-by-tick
        Self::new(0.0001, num_channels)
    }

    /// Create an adaptive delta encoder
    pub fn adaptive(threshold: f64, num_channels: usize, adaptation_rate: f64) -> Self {
        Self {
            threshold,
            previous_values: vec![0.0; num_channels],
            adaptive: true,
            adaptation_rate,
            thresholds: vec![threshold; num_channels],
        }
    }

    /// Reset the encoder state
    pub fn reset(&mut self) {
        self.previous_values = vec![0.0; self.previous_values.len()];
        self.thresholds = vec![self.threshold; self.thresholds.len()];
    }

    /// Set initial values
    pub fn initialize(&mut self, values: &[f64]) {
        assert_eq!(values.len(), self.previous_values.len());
        self.previous_values = values.to_vec();
    }

    /// Get current thresholds
    pub fn thresholds(&self) -> &[f64] {
        &self.thresholds
    }

    /// Process values and return spike events with polarities
    pub fn process(&mut self, values: &[f64], time: f64) -> Vec<Option<(Spike, f64)>> {
        assert_eq!(values.len(), self.previous_values.len());

        let mut results = Vec::with_capacity(values.len());

        for (i, (&current, &previous)) in values.iter()
            .zip(self.previous_values.iter())
            .enumerate()
        {
            let delta = current - previous;
            let threshold = if self.adaptive {
                self.thresholds[i]
            } else {
                self.threshold
            };

            if delta.abs() > threshold {
                let polarity = if delta > 0.0 {
                    SpikePolarity::Positive
                } else {
                    SpikePolarity::Negative
                };

                let spike = Spike {
                    time,
                    polarity,
                    source: Some(i),
                };

                results.push(Some((spike, delta)));

                // Update previous value
                self.previous_values[i] = current;

                // Adapt threshold if enabled
                if self.adaptive {
                    let activity = delta.abs();
                    self.thresholds[i] = self.thresholds[i] * (1.0 - self.adaptation_rate)
                        + activity * self.adaptation_rate;
                }
            } else {
                results.push(None);
            }
        }

        results
    }

    /// Process and return only spikes (for use as input currents)
    pub fn process_to_currents(&mut self, values: &[f64], time: f64) -> Vec<f64> {
        let results = self.process(values, time);

        results.iter().map(|r| {
            match r {
                Some((spike, delta)) => delta.signum() * delta.abs().sqrt() * 10.0,
                None => 0.0,
            }
        }).collect()
    }

    /// Number of channels
    pub fn num_channels(&self) -> usize {
        self.previous_values.len()
    }
}

impl SpikeEncoder for DeltaEncoder {
    fn encode(&self, value: f64) -> f64 {
        // For delta encoding, return the change magnitude
        // This is a simplified version - full encoding needs state
        value.abs()
    }

    fn generate_spikes(&self, values: &[f64], time: f64) -> Vec<Option<Spike>> {
        // Note: This doesn't update state. Use process() for stateful encoding
        values.iter().enumerate().map(|(i, &value)| {
            let previous = self.previous_values.get(i).copied().unwrap_or(0.0);
            let delta = value - previous;

            if delta.abs() > self.threshold {
                Some(Spike {
                    time,
                    polarity: if delta > 0.0 {
                        SpikePolarity::Positive
                    } else {
                        SpikePolarity::Negative
                    },
                    source: Some(i),
                })
            } else {
                None
            }
        }).collect()
    }
}

/// Dual-channel delta encoder for ON/OFF events
///
/// Like biological retina: separate channels for increases and decreases.
#[derive(Debug, Clone)]
pub struct DualChannelDeltaEncoder {
    /// Base delta encoder
    encoder: DeltaEncoder,
}

impl DualChannelDeltaEncoder {
    /// Create a new dual-channel encoder
    pub fn new(threshold: f64, num_inputs: usize) -> Self {
        Self {
            encoder: DeltaEncoder::new(threshold, num_inputs),
        }
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.encoder.reset();
    }

    /// Initialize with values
    pub fn initialize(&mut self, values: &[f64]) {
        self.encoder.initialize(values);
    }

    /// Process values and return ON/OFF spikes
    ///
    /// Returns (ON_spikes, OFF_spikes) where:
    /// - ON_spikes fire for positive changes
    /// - OFF_spikes fire for negative changes
    pub fn process(&mut self, values: &[f64], time: f64) -> (Vec<bool>, Vec<bool>) {
        let results = self.encoder.process(values, time);

        let mut on_spikes = vec![false; values.len()];
        let mut off_spikes = vec![false; values.len()];

        for (i, result) in results.iter().enumerate() {
            if let Some((spike, _)) = result {
                match spike.polarity {
                    SpikePolarity::Positive => on_spikes[i] = true,
                    SpikePolarity::Negative => off_spikes[i] = true,
                }
            }
        }

        (on_spikes, off_spikes)
    }

    /// Get total number of output channels (2x input)
    pub fn num_outputs(&self) -> usize {
        self.encoder.num_channels() * 2
    }

    /// Convert to flat spike vector [ON_0, OFF_0, ON_1, OFF_1, ...]
    pub fn process_flat(&mut self, values: &[f64], time: f64) -> Vec<bool> {
        let (on, off) = self.process(values, time);
        let mut flat = Vec::with_capacity(on.len() * 2);

        for (on_spike, off_spike) in on.into_iter().zip(off.into_iter()) {
            flat.push(on_spike);
            flat.push(off_spike);
        }

        flat
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_encoder() {
        let mut encoder = DeltaEncoder::new(0.1, 3);
        encoder.initialize(&[1.0, 2.0, 3.0]);

        // Small change - no spike
        let results = encoder.process(&[1.05, 2.05, 3.05], 1.0);
        assert!(results.iter().all(|r| r.is_none()));

        // Large change - spike
        let results = encoder.process(&[1.2, 2.05, 3.05], 2.0);
        assert!(results[0].is_some());
        assert!(results[1].is_none());
        assert!(results[2].is_none());
    }

    #[test]
    fn test_delta_polarity() {
        let mut encoder = DeltaEncoder::new(0.1, 1);
        encoder.initialize(&[1.0]);

        // Positive change
        let results = encoder.process(&[1.2], 1.0);
        assert!(matches!(
            results[0],
            Some((Spike { polarity: SpikePolarity::Positive, .. }, _))
        ));

        // Negative change
        let results = encoder.process(&[0.9], 2.0);
        assert!(matches!(
            results[0],
            Some((Spike { polarity: SpikePolarity::Negative, .. }, _))
        ));
    }

    #[test]
    fn test_dual_channel() {
        let mut encoder = DualChannelDeltaEncoder::new(0.1, 2);
        encoder.initialize(&[1.0, 1.0]);

        // Channel 0 increases, channel 1 decreases
        let (on, off) = encoder.process(&[1.2, 0.8], 1.0);

        assert!(on[0]);   // Channel 0 ON
        assert!(!off[0]); // Channel 0 OFF
        assert!(!on[1]);  // Channel 1 ON
        assert!(off[1]);  // Channel 1 OFF
    }

    #[test]
    fn test_adaptive_threshold() {
        let mut encoder = DeltaEncoder::adaptive(0.1, 1, 0.5);
        encoder.initialize(&[1.0]);

        // Large change should increase threshold
        encoder.process(&[2.0], 1.0);

        assert!(encoder.thresholds()[0] > 0.1);
    }
}
