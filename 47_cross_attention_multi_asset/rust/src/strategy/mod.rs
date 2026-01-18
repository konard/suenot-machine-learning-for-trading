//! Strategy module for Cross-Attention Multi-Asset Trading
//!
//! This module provides:
//! - Signal generation from model predictions
//! - Backtesting engine with realistic constraints

mod signals;
mod backtest;

pub use signals::{Signal, SignalGenerator, SignalConfig};
pub use backtest::{Backtest, BacktestConfig, BacktestResult, PerformanceMetrics};

use serde::{Deserialize, Serialize};

/// Trading action
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    /// Buy/long position
    Buy,
    /// Sell/short position
    Sell,
    /// Hold current position
    Hold,
}

/// Portfolio position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Asset symbol
    pub symbol: String,
    /// Position size (positive = long, negative = short)
    pub size: f64,
    /// Entry price
    pub entry_price: f64,
    /// Current unrealized PnL
    pub unrealized_pnl: f64,
}

impl Position {
    /// Create a new position
    pub fn new(symbol: &str, size: f64, entry_price: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            size,
            entry_price,
            unrealized_pnl: 0.0,
        }
    }

    /// Update unrealized PnL
    pub fn update_pnl(&mut self, current_price: f64) {
        self.unrealized_pnl = self.size * (current_price - self.entry_price);
    }

    /// Check if position is long
    pub fn is_long(&self) -> bool {
        self.size > 0.0
    }

    /// Check if position is short
    pub fn is_short(&self) -> bool {
        self.size < 0.0
    }

    /// Get absolute position size
    pub fn abs_size(&self) -> f64 {
        self.size.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position() {
        let mut pos = Position::new("BTCUSDT", 1.0, 50000.0);

        assert!(pos.is_long());
        assert!(!pos.is_short());
        assert_eq!(pos.abs_size(), 1.0);

        pos.update_pnl(51000.0);
        assert_eq!(pos.unrealized_pnl, 1000.0);
    }

    #[test]
    fn test_short_position() {
        let mut pos = Position::new("BTCUSDT", -0.5, 50000.0);

        assert!(!pos.is_long());
        assert!(pos.is_short());
        assert_eq!(pos.abs_size(), 0.5);

        pos.update_pnl(49000.0);
        assert_eq!(pos.unrealized_pnl, 500.0); // Short profits from price decrease
    }
}
