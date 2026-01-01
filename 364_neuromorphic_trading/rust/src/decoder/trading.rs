//! Trading Signal Decoder
//!
//! Decodes SNN output spikes into actionable trading signals.

use super::{Decoder, DecoderConfig, TradingSignal};
use crate::neuron::SpikeEvent;

/// Trading signal decoder using winner-take-all
#[derive(Debug, Clone)]
pub struct TradingDecoder {
    config: DecoderConfig,
}

impl TradingDecoder {
    /// Create a new trading decoder
    pub fn new(config: DecoderConfig) -> Self {
        Self { config }
    }

    /// Count spikes for a specific neuron type
    fn count_spikes_for_type(&self, spikes: &[SpikeEvent], neuron_type: usize) -> usize {
        spikes.iter()
            .filter(|s| s.neuron_id % self.config.output_size == neuron_type)
            .count()
    }

    /// Calculate urgency based on spike timing
    fn calculate_urgency(&self, spikes: &[SpikeEvent], neuron_type: usize) -> f64 {
        let type_spikes: Vec<_> = spikes.iter()
            .filter(|s| s.neuron_id % self.config.output_size == neuron_type)
            .collect();

        if type_spikes.is_empty() {
            return 0.0;
        }

        // Earlier spikes = higher urgency
        let avg_time: f64 = type_spikes.iter().map(|s| s.time).sum::<f64>() / type_spikes.len() as f64;
        let normalized_time = avg_time / self.config.time_window;

        1.0 - normalized_time.clamp(0.0, 1.0)
    }
}

impl Default for TradingDecoder {
    fn default() -> Self {
        Self::new(DecoderConfig::default())
    }
}

impl Decoder for TradingDecoder {
    fn decode(&self, spikes: &[SpikeEvent]) -> TradingSignal {
        // Neuron 0: BUY, Neuron 1: SELL, Neuron 2: HOLD
        let buy_count = self.count_spikes_for_type(spikes, 0);
        let sell_count = self.count_spikes_for_type(spikes, 1);
        let hold_count = self.count_spikes_for_type(spikes, 2);

        let total = buy_count + sell_count + hold_count;

        if total == 0 {
            return TradingSignal::Hold;
        }

        // Winner-take-all
        let (winner, count) = if buy_count >= sell_count && buy_count >= hold_count {
            (0, buy_count)
        } else if sell_count >= buy_count && sell_count >= hold_count {
            (1, sell_count)
        } else {
            (2, hold_count)
        };

        let confidence = count as f64 / total as f64;

        if confidence < self.config.confidence_threshold {
            return TradingSignal::Hold;
        }

        match winner {
            0 => TradingSignal::Buy {
                confidence,
                urgency: self.calculate_urgency(spikes, 0),
            },
            1 => TradingSignal::Sell {
                confidence,
                urgency: self.calculate_urgency(spikes, 1),
            },
            _ => TradingSignal::Hold,
        }
    }

    fn reset(&mut self) {
        // Decoder is stateless
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_buy_signal() {
        let decoder = TradingDecoder::new(DecoderConfig {
            confidence_threshold: 0.4,
            ..Default::default()
        });

        // More BUY spikes
        let spikes = vec![
            SpikeEvent::new(0, 1.0),  // BUY
            SpikeEvent::new(0, 2.0),  // BUY
            SpikeEvent::new(0, 3.0),  // BUY
            SpikeEvent::new(1, 4.0),  // SELL
        ];

        let signal = decoder.decode(&spikes);
        assert!(signal.is_buy());
        assert!(signal.confidence().unwrap() > 0.5);
    }

    #[test]
    fn test_decoder_sell_signal() {
        let decoder = TradingDecoder::new(DecoderConfig {
            confidence_threshold: 0.4,
            ..Default::default()
        });

        // More SELL spikes
        let spikes = vec![
            SpikeEvent::new(1, 1.0),  // SELL
            SpikeEvent::new(1, 2.0),  // SELL
            SpikeEvent::new(1, 3.0),  // SELL
            SpikeEvent::new(0, 4.0),  // BUY
        ];

        let signal = decoder.decode(&spikes);
        assert!(signal.is_sell());
    }

    #[test]
    fn test_decoder_low_confidence_hold() {
        let decoder = TradingDecoder::new(DecoderConfig {
            confidence_threshold: 0.9,
            ..Default::default()
        });

        // Mixed spikes
        let spikes = vec![
            SpikeEvent::new(0, 1.0),
            SpikeEvent::new(1, 2.0),
            SpikeEvent::new(2, 3.0),
        ];

        let signal = decoder.decode(&spikes);
        assert!(signal.is_hold());
    }

    #[test]
    fn test_decoder_no_spikes() {
        let decoder = TradingDecoder::default();
        let signal = decoder.decode(&[]);
        assert!(signal.is_hold());
    }
}
