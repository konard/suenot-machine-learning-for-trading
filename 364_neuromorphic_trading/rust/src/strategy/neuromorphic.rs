//! Neuromorphic Trading Strategy
//!
//! Implements a trading strategy based on spiking neural network outputs.

use super::{Position, StrategyConfig, TradingStrategy};
use crate::decoder::TradingSignal;
use crate::network::NetworkState;

/// Neuromorphic trading strategy
#[derive(Debug, Clone)]
pub struct NeuromorphicStrategy {
    config: StrategyConfig,
}

impl NeuromorphicStrategy {
    /// Create a new neuromorphic strategy
    pub fn new(config: StrategyConfig) -> Self {
        Self { config }
    }

    /// Get the configuration
    pub fn config(&self) -> &StrategyConfig {
        &self.config
    }

    /// Check if network activity is anomalous
    fn is_anomalous_activity(&self, network_state: &NetworkState) -> bool {
        network_state.avg_spike_rate > self.config.spike_rate_threshold
    }
}

impl Default for NeuromorphicStrategy {
    fn default() -> Self {
        Self::new(StrategyConfig::default())
    }
}

impl TradingStrategy for NeuromorphicStrategy {
    fn validate_signal(&self, signal: &TradingSignal, network_state: &NetworkState) -> bool {
        // Reject if network activity is anomalous
        if self.is_anomalous_activity(network_state) {
            return false;
        }

        // Check confidence threshold
        match signal {
            TradingSignal::Buy { confidence, .. } |
            TradingSignal::Sell { confidence, .. } => {
                *confidence >= self.config.confidence_threshold
            }
            TradingSignal::Hold => true,
        }
    }

    fn calculate_position_size(&self, signal: &TradingSignal, current_price: f64) -> f64 {
        match signal {
            TradingSignal::Buy { confidence, urgency } => {
                // Scale position size by confidence and urgency
                let scale = confidence * (0.5 + 0.5 * urgency);
                (self.config.max_position_size * scale).min(self.config.max_position_size)
            }
            TradingSignal::Sell { confidence, urgency } => {
                let scale = confidence * (0.5 + 0.5 * urgency);
                -(self.config.max_position_size * scale).min(self.config.max_position_size)
            }
            TradingSignal::Hold => 0.0,
        }
    }

    fn should_close_position(&self, signal: &TradingSignal, position: &Position) -> bool {
        if position.is_flat() {
            return false;
        }

        // Close long if we get a sell signal
        if position.is_long() && signal.is_sell() {
            return true;
        }

        // Close short if we get a buy signal
        if position.is_short() && signal.is_buy() {
            return true;
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_normal_network_state() -> NetworkState {
        NetworkState {
            avg_membrane_potential: 0.5,
            avg_spike_rate: 50.0,
            spike_count: 10,
            active_neurons: 50,
            current_time: 100.0,
        }
    }

    fn create_anomalous_network_state() -> NetworkState {
        NetworkState {
            avg_spike_rate: 200.0,  // Above threshold
            ..create_normal_network_state()
        }
    }

    #[test]
    fn test_validate_high_confidence() {
        let strategy = NeuromorphicStrategy::default();
        let state = create_normal_network_state();

        let signal = TradingSignal::Buy {
            confidence: 0.8,
            urgency: 0.5,
        };

        assert!(strategy.validate_signal(&signal, &state));
    }

    #[test]
    fn test_validate_low_confidence() {
        let strategy = NeuromorphicStrategy::default();
        let state = create_normal_network_state();

        let signal = TradingSignal::Buy {
            confidence: 0.3,
            urgency: 0.5,
        };

        assert!(!strategy.validate_signal(&signal, &state));
    }

    #[test]
    fn test_validate_anomalous_activity() {
        let strategy = NeuromorphicStrategy::default();
        let state = create_anomalous_network_state();

        let signal = TradingSignal::Buy {
            confidence: 0.9,
            urgency: 0.9,
        };

        assert!(!strategy.validate_signal(&signal, &state));
    }

    #[test]
    fn test_position_size() {
        let strategy = NeuromorphicStrategy::new(StrategyConfig {
            max_position_size: 1.0,
            ..Default::default()
        });

        let buy_signal = TradingSignal::Buy {
            confidence: 0.8,
            urgency: 0.6,
        };

        let size = strategy.calculate_position_size(&buy_signal, 50000.0);
        assert!(size > 0.0);
        assert!(size <= 1.0);
    }

    #[test]
    fn test_should_close_long() {
        let strategy = NeuromorphicStrategy::default();

        let long_position = Position {
            size: 1.0,
            entry_price: 50000.0,
            ..Default::default()
        };

        let sell_signal = TradingSignal::Sell {
            confidence: 0.8,
            urgency: 0.5,
        };

        assert!(strategy.should_close_position(&sell_signal, &long_position));
    }

    #[test]
    fn test_hold_keeps_position() {
        let strategy = NeuromorphicStrategy::default();

        let long_position = Position {
            size: 1.0,
            entry_price: 50000.0,
            ..Default::default()
        };

        assert!(!strategy.should_close_position(&TradingSignal::Hold, &long_position));
    }
}
