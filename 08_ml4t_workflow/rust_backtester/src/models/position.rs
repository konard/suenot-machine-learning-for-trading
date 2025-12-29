//! Position data model.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use super::OrderSide;

/// Represents an open position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Symbol (e.g., "BTCUSDT")
    pub symbol: String,
    /// Position side (Buy = Long, Sell = Short)
    pub side: OrderSide,
    /// Position size
    pub size: f64,
    /// Average entry price
    pub entry_price: f64,
    /// Current unrealized PnL
    pub unrealized_pnl: f64,
    /// Realized PnL
    pub realized_pnl: f64,
    /// Position open time
    pub opened_at: DateTime<Utc>,
    /// Last update time
    pub updated_at: DateTime<Utc>,
}

impl Position {
    /// Create a new position.
    pub fn new(symbol: String, side: OrderSide, size: f64, entry_price: f64) -> Self {
        let now = Utc::now();
        Self {
            symbol,
            side,
            size,
            entry_price,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            opened_at: now,
            updated_at: now,
        }
    }

    /// Check if position is long.
    pub fn is_long(&self) -> bool {
        self.side == OrderSide::Buy
    }

    /// Check if position is short.
    pub fn is_short(&self) -> bool {
        self.side == OrderSide::Sell
    }

    /// Update unrealized PnL based on current price.
    pub fn update_pnl(&mut self, current_price: f64) {
        let price_diff = current_price - self.entry_price;
        self.unrealized_pnl = match self.side {
            OrderSide::Buy => price_diff * self.size,
            OrderSide::Sell => -price_diff * self.size,
        };
        self.updated_at = Utc::now();
    }

    /// Calculate position value at current price.
    pub fn value(&self, current_price: f64) -> f64 {
        self.size * current_price
    }

    /// Calculate return percentage.
    pub fn return_pct(&self, current_price: f64) -> f64 {
        let price_diff = current_price - self.entry_price;
        let raw_return = price_diff / self.entry_price;
        match self.side {
            OrderSide::Buy => raw_return * 100.0,
            OrderSide::Sell => -raw_return * 100.0,
        }
    }

    /// Add to position (average up/down).
    pub fn add(&mut self, size: f64, price: f64) {
        let total_cost = self.entry_price * self.size + price * size;
        self.size += size;
        self.entry_price = total_cost / self.size;
        self.updated_at = Utc::now();
    }

    /// Reduce position and return realized PnL.
    pub fn reduce(&mut self, size: f64, price: f64) -> f64 {
        let reduce_size = size.min(self.size);
        let price_diff = price - self.entry_price;
        let pnl = match self.side {
            OrderSide::Buy => price_diff * reduce_size,
            OrderSide::Sell => -price_diff * reduce_size,
        };

        self.size -= reduce_size;
        self.realized_pnl += pnl;
        self.updated_at = Utc::now();

        pnl
    }

    /// Check if position is closed (size is zero).
    pub fn is_closed(&self) -> bool {
        self.size.abs() < 1e-10
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_long_position_pnl() {
        let mut pos = Position::new("BTCUSDT".to_string(), OrderSide::Buy, 1.0, 50000.0);
        pos.update_pnl(55000.0);
        assert!((pos.unrealized_pnl - 5000.0).abs() < 0.01);
    }

    #[test]
    fn test_short_position_pnl() {
        let mut pos = Position::new("BTCUSDT".to_string(), OrderSide::Sell, 1.0, 50000.0);
        pos.update_pnl(45000.0);
        assert!((pos.unrealized_pnl - 5000.0).abs() < 0.01);
    }

    #[test]
    fn test_reduce_position() {
        let mut pos = Position::new("BTCUSDT".to_string(), OrderSide::Buy, 2.0, 50000.0);
        let pnl = pos.reduce(1.0, 55000.0);
        assert!((pnl - 5000.0).abs() < 0.01);
        assert!((pos.size - 1.0).abs() < 0.01);
    }
}
