//! Temporal Coding Encoder
//!
//! Encodes values as spike times - higher values produce earlier spikes.

use super::{Encoder, EncoderConfig, MarketData, normalize};
use crate::neuron::SpikeEvent;

/// Temporal coding encoder
#[derive(Debug, Clone)]
pub struct TemporalEncoder {
    config: EncoderConfig,
}

impl TemporalEncoder {
    /// Create a new temporal encoder
    pub fn new(config: EncoderConfig) -> Self {
        Self { config }
    }

    /// Encode a single value - higher values spike earlier
    fn encode_value(&self, value: f64, min: f64, max: f64, neuron_id: usize) -> SpikeEvent {
        let normalized = normalize(value, min, max);
        // Higher values = earlier spike (time closer to 0)
        let spike_time = self.config.time_window * (1.0 - normalized);
        SpikeEvent::new(neuron_id, spike_time)
    }
}

impl Default for TemporalEncoder {
    fn default() -> Self {
        Self::new(EncoderConfig::default())
    }
}

impl Encoder for TemporalEncoder {
    fn encode(&self, data: &MarketData) -> Vec<SpikeEvent> {
        let mut spikes = Vec::new();
        let mut neuron_id = 0;

        // Encode bid prices
        for &price in &data.bid_prices {
            spikes.push(self.encode_value(
                price,
                self.config.price_range.0,
                self.config.price_range.1,
                neuron_id,
            ));
            neuron_id += 1;
        }

        // Encode ask prices
        for &price in &data.ask_prices {
            spikes.push(self.encode_value(
                price,
                self.config.price_range.0,
                self.config.price_range.1,
                neuron_id,
            ));
            neuron_id += 1;
        }

        // Encode bid volumes
        for &volume in &data.bid_volumes {
            spikes.push(self.encode_value(
                volume,
                self.config.volume_range.0,
                self.config.volume_range.1,
                neuron_id,
            ));
            neuron_id += 1;
        }

        // Encode ask volumes
        for &volume in &data.ask_volumes {
            spikes.push(self.encode_value(
                volume,
                self.config.volume_range.0,
                self.config.volume_range.1,
                neuron_id,
            ));
            neuron_id += 1;
        }

        spikes
    }

    fn output_size(&self) -> usize {
        // One neuron per feature value
        4 * 8  // 4 features, 8 levels each
    }

    fn reset(&mut self) {
        // Temporal encoder is stateless
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_temporal_encoder() {
        let encoder = TemporalEncoder::default();

        let data = MarketData {
            bid_prices: vec![50000.0],
            ask_prices: vec![50001.0],
            bid_volumes: vec![1.0],
            ask_volumes: vec![1.0],
            timestamp: Utc::now(),
        };

        let spikes = encoder.encode(&data);
        assert_eq!(spikes.len(), 4);  // One spike per feature
    }

    #[test]
    fn test_higher_values_spike_earlier() {
        let encoder = TemporalEncoder::new(EncoderConfig {
            time_window: 10.0,
            ..Default::default()
        });

        let high_data = MarketData {
            bid_prices: vec![90000.0],
            ask_prices: vec![],
            bid_volumes: vec![],
            ask_volumes: vec![],
            timestamp: Utc::now(),
        };

        let low_data = MarketData {
            bid_prices: vec![10000.0],
            ask_prices: vec![],
            bid_volumes: vec![],
            ask_volumes: vec![],
            timestamp: Utc::now(),
        };

        let high_spikes = encoder.encode(&high_data);
        let low_spikes = encoder.encode(&low_data);

        assert!(high_spikes[0].time < low_spikes[0].time);
    }
}
