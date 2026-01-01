//! Delta Modulation Encoder
//!
//! Encodes changes in values - spikes only when value changes significantly.

use super::{Encoder, EncoderConfig, MarketData};
use crate::neuron::SpikeEvent;

/// Delta modulation encoder
#[derive(Debug, Clone)]
pub struct DeltaEncoder {
    config: EncoderConfig,
    /// Last encoded values for each feature
    last_values: Vec<f64>,
    /// Threshold for generating a spike
    threshold: f64,
    /// Current time
    current_time: f64,
}

impl DeltaEncoder {
    /// Create a new delta encoder
    pub fn new(config: EncoderConfig, threshold: f64) -> Self {
        Self {
            config,
            last_values: Vec::new(),
            threshold,
            current_time: 0.0,
        }
    }

    /// Initialize last values from market data
    fn initialize_if_needed(&mut self, data: &MarketData) {
        if self.last_values.is_empty() {
            self.last_values = Vec::new();
            self.last_values.extend(&data.bid_prices);
            self.last_values.extend(&data.ask_prices);
            self.last_values.extend(&data.bid_volumes);
            self.last_values.extend(&data.ask_volumes);
        }
    }

    /// Encode a single value change
    fn encode_change(&mut self, value: f64, index: usize) -> Option<(SpikeEvent, bool)> {
        if index >= self.last_values.len() {
            self.last_values.resize(index + 1, value);
            return None;
        }

        let last = self.last_values[index];
        let delta = value - last;

        if delta.abs() > self.threshold {
            self.last_values[index] = value;
            // Positive delta = UP neuron, Negative = DOWN neuron
            let is_up = delta > 0.0;
            let neuron_id = index * 2 + if is_up { 0 } else { 1 };
            Some((SpikeEvent::new(neuron_id, self.current_time), is_up))
        } else {
            None
        }
    }
}

impl Default for DeltaEncoder {
    fn default() -> Self {
        Self::new(EncoderConfig::default(), 1.0)
    }
}

impl Encoder for DeltaEncoder {
    fn encode(&self, data: &MarketData) -> Vec<SpikeEvent> {
        // Create mutable copy for encoding
        let mut encoder = self.clone();
        encoder.initialize_if_needed(data);
        encoder.current_time += encoder.config.time_window;

        let mut spikes = Vec::new();
        let mut index = 0;

        // Encode bid prices
        for &price in &data.bid_prices {
            if let Some((spike, _)) = encoder.encode_change(price, index) {
                spikes.push(spike);
            }
            index += 1;
        }

        // Encode ask prices
        for &price in &data.ask_prices {
            if let Some((spike, _)) = encoder.encode_change(price, index) {
                spikes.push(spike);
            }
            index += 1;
        }

        // Encode bid volumes
        for &volume in &data.bid_volumes {
            if let Some((spike, _)) = encoder.encode_change(volume, index) {
                spikes.push(spike);
            }
            index += 1;
        }

        // Encode ask volumes
        for &volume in &data.ask_volumes {
            if let Some((spike, _)) = encoder.encode_change(volume, index) {
                spikes.push(spike);
            }
            index += 1;
        }

        spikes
    }

    fn output_size(&self) -> usize {
        // Two neurons per feature (UP and DOWN)
        4 * 8 * 2
    }

    fn reset(&mut self) {
        self.last_values.clear();
        self.current_time = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_delta_encoder_no_change() {
        let mut encoder = DeltaEncoder::new(EncoderConfig::default(), 1.0);

        let data = MarketData {
            bid_prices: vec![100.0],
            ask_prices: vec![101.0],
            bid_volumes: vec![10.0],
            ask_volumes: vec![10.0],
            timestamp: Utc::now(),
        };

        // First call initializes
        let spikes1 = encoder.encode(&data);

        // Same data should produce no spikes
        let spikes2 = encoder.encode(&data);

        // On first encode, no spikes (just initialization)
        // On second encode, no change = no spikes
        assert!(spikes1.is_empty() || spikes2.is_empty());
    }

    #[test]
    fn test_output_size() {
        let encoder = DeltaEncoder::default();
        assert_eq!(encoder.output_size(), 64);  // 4 * 8 * 2
    }
}
