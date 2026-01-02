//! Risk management module

use super::signals::Trade;
use serde::{Deserialize, Serialize};

/// Risk management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskManager {
    /// Stop loss percentage (e.g., 0.02 = 2%)
    pub stop_loss_pct: f32,
    /// Take profit percentage
    pub take_profit_pct: f32,
    /// Maximum daily drawdown before reducing risk
    pub max_daily_drawdown: f32,
    /// Maximum weekly drawdown before stopping
    pub max_weekly_drawdown: f32,
    /// Maximum holding time in milliseconds
    pub max_hold_time: i64,
    /// ATR multiplier for dynamic stop loss
    pub atr_multiplier: f32,
    /// Whether to use trailing stop
    pub use_trailing_stop: bool,
    /// Trailing stop distance (as fraction of price)
    pub trailing_stop_distance: f32,
}

impl Default for RiskManager {
    fn default() -> Self {
        Self {
            stop_loss_pct: 0.02,
            take_profit_pct: 0.03,
            max_daily_drawdown: 0.03,
            max_weekly_drawdown: 0.10,
            max_hold_time: 12 * 60 * 60 * 1000, // 12 hours
            atr_multiplier: 2.0,
            use_trailing_stop: true,
            trailing_stop_distance: 0.015,
        }
    }
}

impl RiskManager {
    /// Create a new risk manager
    pub fn new(stop_loss_pct: f32, take_profit_pct: f32) -> Self {
        Self {
            stop_loss_pct,
            take_profit_pct,
            ..Default::default()
        }
    }

    /// Check if stop loss is triggered
    pub fn check_stop_loss(&self, trade: &Trade, current_price: f32) -> bool {
        let unrealized_return = trade.calculate_pnl(current_price) / trade.size;
        unrealized_return <= -self.stop_loss_pct
    }

    /// Check if take profit is triggered
    pub fn check_take_profit(&self, trade: &Trade, current_price: f32) -> bool {
        let unrealized_return = trade.calculate_pnl(current_price) / trade.size;
        unrealized_return >= self.take_profit_pct
    }

    /// Check if maximum hold time exceeded
    pub fn check_max_hold_time(&self, trade: &Trade, current_time: i64) -> bool {
        (current_time - trade.entry_time) >= self.max_hold_time
    }

    /// Check if position should be closed
    ///
    /// Returns (should_close, reason)
    pub fn should_close(
        &self,
        trade: &Trade,
        current_price: f32,
        current_time: i64,
    ) -> (bool, Option<CloseReason>) {
        if self.check_stop_loss(trade, current_price) {
            return (true, Some(CloseReason::StopLoss));
        }

        if self.check_take_profit(trade, current_price) {
            return (true, Some(CloseReason::TakeProfit));
        }

        if self.check_max_hold_time(trade, current_time) {
            return (true, Some(CloseReason::MaxHoldTime));
        }

        (false, None)
    }

    /// Calculate stop loss price for a trade
    pub fn calculate_stop_loss(&self, entry_price: f32, direction: f32) -> f32 {
        entry_price * (1.0 - self.stop_loss_pct * direction)
    }

    /// Calculate take profit price for a trade
    pub fn calculate_take_profit(&self, entry_price: f32, direction: f32) -> f32 {
        entry_price * (1.0 + self.take_profit_pct * direction)
    }

    /// Calculate dynamic stop loss based on ATR
    pub fn calculate_atr_stop_loss(
        &self,
        entry_price: f32,
        direction: f32,
        atr: f32,
    ) -> f32 {
        entry_price - (self.atr_multiplier * atr * direction)
    }

    /// Update trailing stop
    pub fn update_trailing_stop(
        &self,
        current_stop: f32,
        current_price: f32,
        direction: f32,
    ) -> f32 {
        if !self.use_trailing_stop {
            return current_stop;
        }

        let new_stop = current_price * (1.0 - self.trailing_stop_distance * direction);

        if direction > 0.0 {
            // Long: stop only moves up
            new_stop.max(current_stop)
        } else {
            // Short: stop only moves down
            new_stop.min(current_stop)
        }
    }

    /// Calculate position size adjustment based on drawdown
    pub fn drawdown_adjustment(&self, current_drawdown: f32) -> f32 {
        if current_drawdown >= self.max_weekly_drawdown {
            return 0.0; // Stop trading
        }

        if current_drawdown >= self.max_daily_drawdown {
            return 0.5; // Reduce by 50%
        }

        1.0 // Full size
    }
}

/// Reason for closing a trade
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CloseReason {
    /// Stop loss triggered
    StopLoss,
    /// Take profit triggered
    TakeProfit,
    /// Maximum hold time exceeded
    MaxHoldTime,
    /// Signal reversed
    SignalReverse,
    /// Manual close
    Manual,
    /// Trailing stop triggered
    TrailingStop,
}

/// Portfolio state for risk tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioState {
    /// Current portfolio value
    pub value: f32,
    /// Initial portfolio value
    pub initial_value: f32,
    /// Peak portfolio value
    pub peak_value: f32,
    /// Daily starting value
    pub daily_start_value: f32,
    /// Weekly starting value
    pub weekly_start_value: f32,
    /// Current drawdown
    pub current_drawdown: f32,
    /// Maximum drawdown
    pub max_drawdown: f32,
    /// Number of trades
    pub num_trades: usize,
    /// Number of winning trades
    pub winning_trades: usize,
    /// Total profit
    pub total_profit: f32,
    /// Total loss
    pub total_loss: f32,
}

impl PortfolioState {
    /// Create a new portfolio state
    pub fn new(initial_value: f32) -> Self {
        Self {
            value: initial_value,
            initial_value,
            peak_value: initial_value,
            daily_start_value: initial_value,
            weekly_start_value: initial_value,
            current_drawdown: 0.0,
            max_drawdown: 0.0,
            num_trades: 0,
            winning_trades: 0,
            total_profit: 0.0,
            total_loss: 0.0,
        }
    }

    /// Update portfolio after a trade
    pub fn update(&mut self, pnl: f32) {
        self.value += pnl;
        self.num_trades += 1;

        if pnl > 0.0 {
            self.winning_trades += 1;
            self.total_profit += pnl;
        } else {
            self.total_loss -= pnl; // Make positive
        }

        // Update peak and drawdown
        if self.value > self.peak_value {
            self.peak_value = self.value;
        }

        self.current_drawdown = (self.peak_value - self.value) / self.peak_value;
        self.max_drawdown = self.max_drawdown.max(self.current_drawdown);
    }

    /// Reset daily tracking
    pub fn reset_daily(&mut self) {
        self.daily_start_value = self.value;
    }

    /// Reset weekly tracking
    pub fn reset_weekly(&mut self) {
        self.weekly_start_value = self.value;
    }

    /// Get daily return
    pub fn daily_return(&self) -> f32 {
        (self.value - self.daily_start_value) / self.daily_start_value
    }

    /// Get weekly return
    pub fn weekly_return(&self) -> f32 {
        (self.value - self.weekly_start_value) / self.weekly_start_value
    }

    /// Get total return
    pub fn total_return(&self) -> f32 {
        (self.value - self.initial_value) / self.initial_value
    }

    /// Get win rate
    pub fn win_rate(&self) -> f32 {
        if self.num_trades == 0 {
            return 0.0;
        }
        self.winning_trades as f32 / self.num_trades as f32
    }

    /// Get profit factor
    pub fn profit_factor(&self) -> f32 {
        if self.total_loss == 0.0 {
            return if self.total_profit > 0.0 {
                f32::INFINITY
            } else {
                0.0
            };
        }
        self.total_profit / self.total_loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stop_loss_check() {
        let risk = RiskManager::default();
        let trade = Trade::open(
            1000,
            50000.0,
            1.0,
            10000.0,
            super::super::signals::TradingSignal::Long,
            0.7,
        );

        // Price drops 3% - should trigger stop loss
        assert!(risk.check_stop_loss(&trade, 48500.0));

        // Price drops 1% - should not trigger
        assert!(!risk.check_stop_loss(&trade, 49500.0));
    }

    #[test]
    fn test_take_profit_check() {
        let risk = RiskManager::default();
        let trade = Trade::open(
            1000,
            50000.0,
            1.0,
            10000.0,
            super::super::signals::TradingSignal::Long,
            0.7,
        );

        // Price rises 4% - should trigger take profit
        assert!(risk.check_take_profit(&trade, 52000.0));

        // Price rises 2% - should not trigger
        assert!(!risk.check_take_profit(&trade, 51000.0));
    }

    #[test]
    fn test_trailing_stop() {
        let risk = RiskManager::default();

        // Initial stop at 49000 (2% below 50000)
        let initial_stop = 49000.0;

        // Price moves to 51000 - stop should move up
        let new_stop = risk.update_trailing_stop(initial_stop, 51000.0, 1.0);
        assert!(new_stop > initial_stop);

        // Price drops - stop should not move down
        let same_stop = risk.update_trailing_stop(new_stop, 50000.0, 1.0);
        assert!((same_stop - new_stop).abs() < 0.01);
    }

    #[test]
    fn test_portfolio_state() {
        let mut portfolio = PortfolioState::new(100000.0);

        // Win a trade
        portfolio.update(1000.0);
        assert_eq!(portfolio.value, 101000.0);
        assert_eq!(portfolio.winning_trades, 1);

        // Lose a trade
        portfolio.update(-500.0);
        assert_eq!(portfolio.value, 100500.0);
        assert!(portfolio.current_drawdown > 0.0);

        assert!((portfolio.win_rate() - 0.5).abs() < 0.01);
    }
}
