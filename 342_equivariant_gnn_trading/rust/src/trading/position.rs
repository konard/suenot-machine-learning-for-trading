//! Position Management

use serde::{Deserialize, Serialize};
use super::signal::TradeDirection;

/// An open trading position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Asset symbol
    pub symbol: String,
    /// Direction (Long or Short)
    pub direction: TradeDirection,
    /// Position size (fraction of capital)
    pub size: f64,
    /// Entry price
    pub entry_price: f64,
    /// Entry timestamp
    pub entry_time: u64,
    /// Current unrealized PnL
    pub unrealized_pnl: f64,
}

impl Position {
    /// Create a new position
    pub fn new(
        symbol: String,
        direction: TradeDirection,
        size: f64,
        entry_price: f64,
        entry_time: u64,
    ) -> Self {
        Self {
            symbol, direction, size, entry_price, entry_time, unrealized_pnl: 0.0,
        }
    }

    /// Update unrealized PnL based on current price
    pub fn update_pnl(&mut self, current_price: f64) {
        let price_change = (current_price - self.entry_price) / self.entry_price;
        self.unrealized_pnl = match self.direction {
            TradeDirection::Long => price_change * self.size,
            TradeDirection::Short => -price_change * self.size,
            TradeDirection::Hold => 0.0,
        };
    }

    /// Calculate PnL if closed at given price
    pub fn calculate_close_pnl(&self, close_price: f64, fee_rate: f64) -> f64 {
        let price_change = (close_price - self.entry_price) / self.entry_price;
        let gross_pnl = match self.direction {
            TradeDirection::Long => price_change * self.size,
            TradeDirection::Short => -price_change * self.size,
            TradeDirection::Hold => 0.0,
        };
        gross_pnl - (fee_rate * self.size * 2.0) // Entry + exit fees
    }
}
