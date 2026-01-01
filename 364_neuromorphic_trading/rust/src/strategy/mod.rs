//! Trading Strategy Module
//!
//! Implements neuromorphic trading strategies using spiking neural networks.

pub mod neuromorphic;

use crate::decoder::TradingSignal;
use crate::network::NetworkState;

/// Strategy configuration
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Minimum confidence threshold for signals
    pub confidence_threshold: f64,
    /// Maximum position size (in base currency)
    pub max_position_size: f64,
    /// Maximum spike rate threshold (for anomaly detection)
    pub spike_rate_threshold: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.6,
            max_position_size: 0.01,
            spike_rate_threshold: 100.0,
        }
    }
}

/// Position information
#[derive(Debug, Clone, Default)]
pub struct Position {
    /// Current position size (positive = long, negative = short)
    pub size: f64,
    /// Average entry price
    pub entry_price: f64,
    /// Unrealized P&L
    pub unrealized_pnl: f64,
    /// Realized P&L
    pub realized_pnl: f64,
}

impl Position {
    /// Check if position is long
    pub fn is_long(&self) -> bool {
        self.size > 0.0
    }

    /// Check if position is short
    pub fn is_short(&self) -> bool {
        self.size < 0.0
    }

    /// Check if position is flat
    pub fn is_flat(&self) -> bool {
        self.size.abs() < 1e-10
    }

    /// Update unrealized P&L
    pub fn update_pnl(&mut self, current_price: f64) {
        if !self.is_flat() {
            self.unrealized_pnl = (current_price - self.entry_price) * self.size;
        }
    }
}

/// Trading strategy trait
pub trait TradingStrategy: Send + Sync {
    /// Validate a trading signal
    fn validate_signal(&self, signal: &TradingSignal, network_state: &NetworkState) -> bool;

    /// Calculate position size for a signal
    fn calculate_position_size(&self, signal: &TradingSignal, current_price: f64) -> f64;

    /// Should we close the current position?
    fn should_close_position(&self, signal: &TradingSignal, position: &Position) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_default() {
        let pos = Position::default();
        assert!(pos.is_flat());
        assert!(!pos.is_long());
        assert!(!pos.is_short());
    }

    #[test]
    fn test_position_long() {
        let pos = Position {
            size: 1.0,
            entry_price: 50000.0,
            ..Default::default()
        };
        assert!(pos.is_long());
        assert!(!pos.is_short());
    }

    #[test]
    fn test_position_pnl() {
        let mut pos = Position {
            size: 1.0,
            entry_price: 50000.0,
            ..Default::default()
        };

        pos.update_pnl(51000.0);
        assert_eq!(pos.unrealized_pnl, 1000.0);

        pos.update_pnl(49000.0);
        assert_eq!(pos.unrealized_pnl, -1000.0);
    }
}
