//! Position management

use serde::{Deserialize, Serialize};

/// Position side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionSide {
    Long,
    Short,
    Flat,
}

/// Trading position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Symbol
    pub symbol: String,
    /// Position side
    pub side: PositionSide,
    /// Position size
    pub size: f64,
    /// Entry price
    pub entry_price: f64,
    /// Entry timestamp
    pub entry_time: i64,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
    /// Realized PnL
    pub realized_pnl: f64,
}

impl Position {
    /// Create new flat position
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            side: PositionSide::Flat,
            size: 0.0,
            entry_price: 0.0,
            entry_time: 0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
        }
    }

    /// Open position
    pub fn open(&mut self, side: PositionSide, size: f64, price: f64, timestamp: i64) {
        self.side = side;
        self.size = size;
        self.entry_price = price;
        self.entry_time = timestamp;
        self.unrealized_pnl = 0.0;
    }

    /// Close position
    pub fn close(&mut self, exit_price: f64) -> f64 {
        let pnl = self.calculate_pnl(exit_price);
        self.realized_pnl += pnl;
        self.side = PositionSide::Flat;
        self.size = 0.0;
        self.entry_price = 0.0;
        self.unrealized_pnl = 0.0;
        pnl
    }

    /// Update unrealized PnL
    pub fn update_pnl(&mut self, current_price: f64) {
        self.unrealized_pnl = self.calculate_pnl(current_price);
    }

    /// Calculate PnL at given price
    pub fn calculate_pnl(&self, price: f64) -> f64 {
        match self.side {
            PositionSide::Long => (price - self.entry_price) * self.size,
            PositionSide::Short => (self.entry_price - price) * self.size,
            PositionSide::Flat => 0.0,
        }
    }

    /// Calculate return percentage
    pub fn calculate_return(&self, price: f64) -> f64 {
        if self.entry_price == 0.0 {
            return 0.0;
        }

        match self.side {
            PositionSide::Long => (price - self.entry_price) / self.entry_price,
            PositionSide::Short => (self.entry_price - price) / self.entry_price,
            PositionSide::Flat => 0.0,
        }
    }

    /// Check if position is open
    pub fn is_open(&self) -> bool {
        self.side != PositionSide::Flat && self.size > 0.0
    }
}

/// Position manager with risk controls
pub struct PositionManager {
    /// Current position
    position: Position,
    /// Maximum position size
    max_position_size: f64,
    /// Maximum drawdown allowed
    max_drawdown: f64,
    /// Stop loss percentage
    stop_loss_pct: f64,
    /// Take profit percentage
    take_profit_pct: f64,
    /// Trailing stop percentage
    trailing_stop_pct: Option<f64>,
    /// Highest price since entry (for trailing stop)
    highest_since_entry: f64,
    /// Lowest price since entry (for trailing stop)
    lowest_since_entry: f64,
    /// Current capital
    capital: f64,
    /// Initial capital
    initial_capital: f64,
    /// Peak capital (for drawdown)
    peak_capital: f64,
}

impl PositionManager {
    /// Create new position manager
    pub fn new(symbol: &str, initial_capital: f64) -> Self {
        Self {
            position: Position::new(symbol),
            max_position_size: initial_capital,
            max_drawdown: 0.2, // 20%
            stop_loss_pct: 0.02, // 2%
            take_profit_pct: 0.04, // 4%
            trailing_stop_pct: None,
            highest_since_entry: 0.0,
            lowest_since_entry: f64::MAX,
            capital: initial_capital,
            initial_capital,
            peak_capital: initial_capital,
        }
    }

    /// Set stop loss
    pub fn with_stop_loss(mut self, pct: f64) -> Self {
        self.stop_loss_pct = pct;
        self
    }

    /// Set take profit
    pub fn with_take_profit(mut self, pct: f64) -> Self {
        self.take_profit_pct = pct;
        self
    }

    /// Set trailing stop
    pub fn with_trailing_stop(mut self, pct: f64) -> Self {
        self.trailing_stop_pct = Some(pct);
        self
    }

    /// Set max drawdown
    pub fn with_max_drawdown(mut self, pct: f64) -> Self {
        self.max_drawdown = pct;
        self
    }

    /// Set max position size
    pub fn with_max_position(mut self, size: f64) -> Self {
        self.max_position_size = size;
        self
    }

    /// Get current position
    pub fn position(&self) -> &Position {
        &self.position
    }

    /// Get current capital
    pub fn capital(&self) -> f64 {
        self.capital
    }

    /// Calculate position size based on risk
    pub fn calculate_position_size(&self, price: f64, volatility: f64) -> f64 {
        // Risk-based sizing: risk capital based on volatility
        let risk_capital = self.capital * 0.01; // Risk 1% per trade
        let position_risk = price * volatility; // Risk per unit

        let size = if position_risk > 0.0 {
            risk_capital / position_risk
        } else {
            0.0
        };

        // Cap at max position
        size.min(self.max_position_size / price)
    }

    /// Open position with risk checks
    pub fn open_position(
        &mut self,
        side: PositionSide,
        size: f64,
        price: f64,
        timestamp: i64,
    ) -> bool {
        // Check drawdown limit
        let current_drawdown = (self.peak_capital - self.capital) / self.peak_capital;
        if current_drawdown >= self.max_drawdown {
            log::warn!("Max drawdown reached: {:.2}%", current_drawdown * 100.0);
            return false;
        }

        // Check if we have enough capital
        let required = size * price;
        if required > self.capital {
            log::warn!("Insufficient capital: need {}, have {}", required, self.capital);
            return false;
        }

        // Close existing position if any
        if self.position.is_open() {
            self.close_position(price);
        }

        // Open new position
        let actual_size = size.min(self.max_position_size / price);
        self.position.open(side, actual_size, price, timestamp);

        // Reset trailing stop tracking
        self.highest_since_entry = price;
        self.lowest_since_entry = price;

        true
    }

    /// Close current position
    pub fn close_position(&mut self, price: f64) -> f64 {
        if !self.position.is_open() {
            return 0.0;
        }

        let pnl = self.position.close(price);
        self.capital += pnl;
        self.peak_capital = self.peak_capital.max(self.capital);

        pnl
    }

    /// Update with new price and check exit conditions
    pub fn update(&mut self, price: f64) -> Option<&str> {
        if !self.position.is_open() {
            return None;
        }

        // Update tracking
        self.highest_since_entry = self.highest_since_entry.max(price);
        self.lowest_since_entry = self.lowest_since_entry.min(price);

        // Update PnL
        self.position.update_pnl(price);

        // Check exit conditions
        let return_pct = self.position.calculate_return(price);

        // Stop loss
        if return_pct <= -self.stop_loss_pct {
            return Some("stop_loss");
        }

        // Take profit
        if return_pct >= self.take_profit_pct {
            return Some("take_profit");
        }

        // Trailing stop
        if let Some(trailing_pct) = self.trailing_stop_pct {
            match self.position.side {
                PositionSide::Long => {
                    let stop_price = self.highest_since_entry * (1.0 - trailing_pct);
                    if price <= stop_price {
                        return Some("trailing_stop");
                    }
                }
                PositionSide::Short => {
                    let stop_price = self.lowest_since_entry * (1.0 + trailing_pct);
                    if price >= stop_price {
                        return Some("trailing_stop");
                    }
                }
                _ => {}
            }
        }

        None
    }

    /// Get current drawdown
    pub fn current_drawdown(&self) -> f64 {
        (self.peak_capital - self.capital) / self.peak_capital
    }

    /// Get total return
    pub fn total_return(&self) -> f64 {
        (self.capital - self.initial_capital) / self.initial_capital
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_pnl() {
        let mut pos = Position::new("BTCUSDT");

        // Long position
        pos.open(PositionSide::Long, 1.0, 100.0, 0);
        assert_eq!(pos.calculate_pnl(110.0), 10.0);
        assert_eq!(pos.calculate_pnl(90.0), -10.0);

        pos.close(110.0);
        assert_eq!(pos.realized_pnl, 10.0);

        // Short position
        pos.open(PositionSide::Short, 1.0, 100.0, 0);
        assert_eq!(pos.calculate_pnl(90.0), 10.0);
        assert_eq!(pos.calculate_pnl(110.0), -10.0);
    }

    #[test]
    fn test_position_manager() {
        let mut pm = PositionManager::new("BTCUSDT", 10000.0)
            .with_stop_loss(0.02)
            .with_take_profit(0.04);

        // Open long
        assert!(pm.open_position(PositionSide::Long, 0.1, 100.0, 0));
        assert!(pm.position().is_open());

        // Update price - no exit
        assert!(pm.update(101.0).is_none());

        // Take profit hit
        assert_eq!(pm.update(105.0), Some("take_profit"));
    }

    #[test]
    fn test_drawdown_limit() {
        let mut pm = PositionManager::new("BTCUSDT", 10000.0)
            .with_max_drawdown(0.1);

        // Simulate losses
        pm.capital = 8500.0; // 15% loss

        // Should not allow new positions
        assert!(!pm.open_position(PositionSide::Long, 0.1, 100.0, 0));
    }
}
