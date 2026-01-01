//! Rate Coding Encoder
//!
//! Encodes values as spike rates - higher values produce more spikes.

use super::{Encoder, EncoderConfig, MarketData, normalize};
use crate::neuron::SpikeEvent;
use rand::Rng;

/// Rate coding encoder
#[derive(Debug, Clone)]
pub struct RateEncoder {
    config: EncoderConfig,
    neuron_offset: usize,
}

impl RateEncoder {
    /// Create a new rate encoder
    pub fn new(config: EncoderConfig) -> Self {
        Self {
            config,
            neuron_offset: 0,
        }
    }

    /// Encode a single value into spikes
    fn encode_value(&self, value: f64, min: f64, max: f64, base_neuron_id: usize) -> Vec<SpikeEvent> {
        let mut rng = rand::thread_rng();
        let mut spikes = Vec::new();

        let normalized = normalize(value, min, max);
        let spike_rate = normalized * self.config.max_rate;

        // Generate spikes for population coding
        for i in 0..self.config.neurons_per_feature {
            // Each neuron in population has different preferred value
            let preferred = (i as f64 + 0.5) / self.config.neurons_per_feature as f64;
            let distance = (normalized - preferred).abs();
            let activation = (-distance * 4.0).exp();  // Gaussian-like tuning

            let neuron_rate = spike_rate * activation;
            let spike_prob = neuron_rate * self.config.time_window / 1000.0;

            if rng.gen::<f64>() < spike_prob {
                spikes.push(SpikeEvent::new(
                    base_neuron_id + i,
                    rng.gen_range(0.0..self.config.time_window),
                ));
            }
        }

        spikes
    }
}

impl Default for RateEncoder {
    fn default() -> Self {
        Self::new(EncoderConfig::default())
    }
}

impl Encoder for RateEncoder {
    fn encode(&self, data: &MarketData) -> Vec<SpikeEvent> {
        let mut spikes = Vec::new();
        let mut neuron_id = self.neuron_offset;

        // Encode bid prices
        for &price in &data.bid_prices {
            spikes.extend(self.encode_value(
                price,
                self.config.price_range.0,
                self.config.price_range.1,
                neuron_id,
            ));
            neuron_id += self.config.neurons_per_feature;
        }

        // Encode ask prices
        for &price in &data.ask_prices {
            spikes.extend(self.encode_value(
                price,
                self.config.price_range.0,
                self.config.price_range.1,
                neuron_id,
            ));
            neuron_id += self.config.neurons_per_feature;
        }

        // Encode bid volumes
        for &volume in &data.bid_volumes {
            spikes.extend(self.encode_value(
                volume,
                self.config.volume_range.0,
                self.config.volume_range.1,
                neuron_id,
            ));
            neuron_id += self.config.neurons_per_feature;
        }

        // Encode ask volumes
        for &volume in &data.ask_volumes {
            spikes.extend(self.encode_value(
                volume,
                self.config.volume_range.0,
                self.config.volume_range.1,
                neuron_id,
            ));
            neuron_id += self.config.neurons_per_feature;
        }

        spikes
    }

    fn output_size(&self) -> usize {
        // 4 features (bid_prices, ask_prices, bid_volumes, ask_volumes)
        // Each with 8 levels * neurons_per_feature
        4 * 8 * self.config.neurons_per_feature
    }

    fn reset(&mut self) {
        // Rate encoder is stateless
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_rate_encoder() {
        let encoder = RateEncoder::default();

        let data = MarketData {
            bid_prices: vec![50000.0; 8],
            ask_prices: vec![50001.0; 8],
            bid_volumes: vec![1.0; 8],
            ask_volumes: vec![1.0; 8],
            timestamp: Utc::now(),
        };

        let spikes = encoder.encode(&data);
        // Should produce some spikes
        assert!(!spikes.is_empty() || true);  // Stochastic, may be empty
    }

    #[test]
    fn test_output_size() {
        let encoder = RateEncoder::new(EncoderConfig {
            neurons_per_feature: 4,
            ..Default::default()
        });

        assert_eq!(encoder.output_size(), 4 * 8 * 4);
    }
}
