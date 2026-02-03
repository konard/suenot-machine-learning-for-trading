//! Trading strategy implementation.

use chrono::{DateTime, Utc};

use super::Signal;

/// An open or closed position.
#[derive(Debug, Clone)]
pub struct Position {
    /// Asset symbol
    pub symbol: String,
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
    /// Entry price
    pub entry_price: f64,
    /// Position direction (1 for long, -1 for short)
    pub direction: i32,
    /// Position size in units
    pub size: f64,
    /// Exit timestamp (if closed)
    pub exit_time: Option<DateTime<Utc>>,
    /// Exit price (if closed)
    pub exit_price: Option<f64>,
    /// Profit/loss (if closed)
    pub pnl: Option<f64>,
    /// Exit reason
    pub exit_reason: Option<String>,
}

impl Position {
    /// Create a new open position.
    pub fn new(
        symbol: String,
        entry_time: DateTime<Utc>,
        entry_price: f64,
        direction: i32,
        size: f64,
    ) -> Self {
        Self {
            symbol,
            entry_time,
            entry_price,
            direction,
            size,
            exit_time: None,
            exit_price: None,
            pnl: None,
            exit_reason: None,
        }
    }

    /// Close the position.
    pub fn close(
        &mut self,
        exit_time: DateTime<Utc>,
        exit_price: f64,
        reason: &str,
    ) {
        let price_change = (exit_price - self.entry_price) / self.entry_price;
        let pnl = self.direction as f64 * price_change * self.size * self.entry_price;

        self.exit_time = Some(exit_time);
        self.exit_price = Some(exit_price);
        self.pnl = Some(pnl);
        self.exit_reason = Some(reason.to_string());
    }

    /// Check if position is open.
    pub fn is_open(&self) -> bool {
        self.exit_time.is_none()
    }

    /// Compute unrealized PnL at current price.
    pub fn unrealized_pnl(&self, current_price: f64) -> f64 {
        let price_change = (current_price - self.entry_price) / self.entry_price;
        self.direction as f64 * price_change * self.size * self.entry_price
    }

    /// Compute holding period in days.
    pub fn holding_days(&self, current_time: DateTime<Utc>) -> i64 {
        let end_time = self.exit_time.unwrap_or(current_time);
        (end_time - self.entry_time).num_days()
    }
}

/// Trading strategy configuration.
#[derive(Debug, Clone)]
pub struct TradingStrategy {
    /// Maximum position size as fraction of portfolio
    pub max_position_size: f64,
    /// Stop-loss as fraction of entry price
    pub stop_loss: f64,
    /// Take-profit as fraction of entry price
    pub take_profit: f64,
    /// Maximum holding period in days
    pub holding_period: i64,
    /// Maximum number of concurrent positions
    pub max_concurrent: usize,
    /// Minimum pre-treatment window in days
    pub min_pre_window: i64,
    /// Transaction cost as fraction of trade value
    pub transaction_cost: f64,
    /// Slippage as fraction of price
    pub slippage: f64,
}

impl Default for TradingStrategy {
    fn default() -> Self {
        Self {
            max_position_size: 0.15,
            stop_loss: 0.10,
            take_profit: 0.15,
            holding_period: 10,
            max_concurrent: 5,
            min_pre_window: 30,
            transaction_cost: 0.001,
            slippage: 0.0005,
        }
    }
}

impl TradingStrategy {
    /// Create a new strategy with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum position size.
    pub fn with_max_position(mut self, size: f64) -> Self {
        self.max_position_size = size;
        self
    }

    /// Set stop-loss level.
    pub fn with_stop_loss(mut self, sl: f64) -> Self {
        self.stop_loss = sl;
        self
    }

    /// Set take-profit level.
    pub fn with_take_profit(mut self, tp: f64) -> Self {
        self.take_profit = tp;
        self
    }

    /// Set holding period.
    pub fn with_holding_period(mut self, days: i64) -> Self {
        self.holding_period = days;
        self
    }

    /// Check if a new position should be opened.
    pub fn should_enter(&self, signal: Signal, current_positions: usize) -> bool {
        if current_positions >= self.max_concurrent {
            return false;
        }

        matches!(signal, Signal::Long | Signal::Short)
    }

    /// Check exit conditions for a position.
    pub fn check_exit(
        &self,
        position: &Position,
        current_price: f64,
        current_time: DateTime<Utc>,
    ) -> Option<String> {
        // Check holding period
        if position.holding_days(current_time) >= self.holding_period {
            return Some("holding_period".to_string());
        }

        let price_change = (current_price - position.entry_price) / position.entry_price;
        let directional_change = position.direction as f64 * price_change;

        // Check stop-loss
        if directional_change < -self.stop_loss {
            return Some("stop_loss".to_string());
        }

        // Check take-profit
        if directional_change > self.take_profit {
            return Some("take_profit".to_string());
        }

        None
    }

    /// Apply slippage to price for order execution.
    pub fn apply_slippage(&self, price: f64, direction: i32) -> f64 {
        if direction > 0 {
            // Long: pay slightly more on entry
            price * (1.0 + self.slippage)
        } else {
            // Short: receive slightly less on entry
            price * (1.0 - self.slippage)
        }
    }

    /// Calculate transaction cost for a trade.
    pub fn calc_transaction_cost(&self, trade_value: f64) -> f64 {
        trade_value * self.transaction_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_pnl() {
        let mut pos = Position::new(
            "BTCUSDT".to_string(),
            Utc::now() - chrono::Duration::days(5),
            40000.0,
            1, // Long
            0.5,
        );

        // Check unrealized PnL at higher price
        let unrealized = pos.unrealized_pnl(44000.0);
        assert!((unrealized - 2000.0).abs() < 0.01);

        // Close position
        pos.close(Utc::now(), 44000.0, "take_profit");
        assert_eq!(pos.pnl, Some(2000.0));
    }

    #[test]
    fn test_strategy_exit_conditions() {
        let strategy = TradingStrategy::new()
            .with_stop_loss(0.05)
            .with_take_profit(0.10)
            .with_holding_period(5);

        let now = Utc::now();
        let entry_time = now - chrono::Duration::days(3);

        let pos = Position::new("ETHUSDT".to_string(), entry_time, 2500.0, 1, 1.0);

        // No exit at small move
        assert!(strategy.check_exit(&pos, 2550.0, now).is_none());

        // Stop-loss trigger
        assert_eq!(
            strategy.check_exit(&pos, 2300.0, now),
            Some("stop_loss".to_string())
        );

        // Take-profit trigger
        assert_eq!(
            strategy.check_exit(&pos, 2800.0, now),
            Some("take_profit".to_string())
        );

        // Holding period trigger
        let old_pos = Position::new(
            "ETHUSDT".to_string(),
            now - chrono::Duration::days(10),
            2500.0,
            1,
            1.0,
        );
        assert_eq!(
            strategy.check_exit(&old_pos, 2500.0, now),
            Some("holding_period".to_string())
        );
    }
}
