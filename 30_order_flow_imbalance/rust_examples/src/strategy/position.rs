//! # Position Manager
//!
//! Position tracking and risk management.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Position side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionSide {
    Long,
    Short,
    Flat,
}

/// Active position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Symbol
    pub symbol: String,
    /// Side
    pub side: PositionSide,
    /// Size
    pub size: f64,
    /// Entry price
    pub entry_price: f64,
    /// Entry time
    pub entry_time: DateTime<Utc>,
    /// Stop loss
    pub stop_loss: Option<f64>,
    /// Take profit
    pub take_profit: Option<f64>,
    /// Unrealized P&L
    pub unrealized_pnl: f64,
}

impl Position {
    /// Create a new position
    pub fn new(
        symbol: String,
        side: PositionSide,
        size: f64,
        entry_price: f64,
        stop_loss: Option<f64>,
        take_profit: Option<f64>,
    ) -> Self {
        Self {
            symbol,
            side,
            size,
            entry_price,
            entry_time: Utc::now(),
            stop_loss,
            take_profit,
            unrealized_pnl: 0.0,
        }
    }

    /// Update with current price
    pub fn update_pnl(&mut self, current_price: f64) {
        let price_diff = current_price - self.entry_price;
        self.unrealized_pnl = match self.side {
            PositionSide::Long => price_diff * self.size,
            PositionSide::Short => -price_diff * self.size,
            PositionSide::Flat => 0.0,
        };
    }

    /// Check if stop loss hit
    pub fn is_stopped_out(&self, current_price: f64) -> bool {
        if let Some(stop) = self.stop_loss {
            match self.side {
                PositionSide::Long => current_price <= stop,
                PositionSide::Short => current_price >= stop,
                PositionSide::Flat => false,
            }
        } else {
            false
        }
    }

    /// Check if take profit hit
    pub fn is_target_hit(&self, current_price: f64) -> bool {
        if let Some(target) = self.take_profit {
            match self.side {
                PositionSide::Long => current_price >= target,
                PositionSide::Short => current_price <= target,
                PositionSide::Flat => false,
            }
        } else {
            false
        }
    }

    /// Get return percentage
    pub fn return_pct(&self) -> f64 {
        if self.entry_price > 0.0 {
            self.unrealized_pnl / (self.entry_price * self.size) * 100.0
        } else {
            0.0
        }
    }

    /// Get holding duration
    pub fn holding_duration(&self) -> chrono::Duration {
        Utc::now() - self.entry_time
    }
}

/// Position manager
pub struct PositionManager {
    /// Current position
    current_position: Option<Position>,
    /// Maximum position size
    max_size: f64,
    /// Maximum holding time (seconds)
    max_holding_time: i64,
    /// Maximum daily loss
    max_daily_loss: f64,
    /// Daily P&L
    daily_pnl: f64,
    /// Closed trades
    closed_trades: Vec<ClosedTrade>,
}

/// Closed trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosedTrade {
    pub symbol: String,
    pub side: PositionSide,
    pub size: f64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub entry_time: DateTime<Utc>,
    pub exit_time: DateTime<Utc>,
    pub pnl: f64,
    pub exit_reason: ExitReason,
}

/// Reason for exiting position
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExitReason {
    StopLoss,
    TakeProfit,
    Signal,
    Timeout,
    EndOfDay,
    Manual,
}

impl PositionManager {
    /// Create new manager
    pub fn new(max_size: f64, max_holding_time: i64, max_daily_loss: f64) -> Self {
        Self {
            current_position: None,
            max_size,
            max_holding_time,
            max_daily_loss,
            daily_pnl: 0.0,
            closed_trades: Vec::new(),
        }
    }

    /// Check if can open position
    pub fn can_open_position(&self, size: f64) -> bool {
        // No existing position
        if self.current_position.is_some() {
            return false;
        }

        // Size within limits
        if size > self.max_size {
            return false;
        }

        // Daily loss limit not hit
        if self.daily_pnl <= -self.max_daily_loss {
            return false;
        }

        true
    }

    /// Open a position
    pub fn open_position(&mut self, position: Position) -> bool {
        if !self.can_open_position(position.size) {
            return false;
        }

        self.current_position = Some(position);
        true
    }

    /// Close current position
    pub fn close_position(&mut self, exit_price: f64, reason: ExitReason) -> Option<ClosedTrade> {
        if let Some(mut pos) = self.current_position.take() {
            pos.update_pnl(exit_price);

            let trade = ClosedTrade {
                symbol: pos.symbol,
                side: pos.side,
                size: pos.size,
                entry_price: pos.entry_price,
                exit_price,
                entry_time: pos.entry_time,
                exit_time: Utc::now(),
                pnl: pos.unrealized_pnl,
                exit_reason: reason,
            };

            self.daily_pnl += trade.pnl;
            self.closed_trades.push(trade.clone());

            Some(trade)
        } else {
            None
        }
    }

    /// Update position with current price
    pub fn update(&mut self, current_price: f64) -> Option<ExitReason> {
        if let Some(pos) = &mut self.current_position {
            pos.update_pnl(current_price);

            // Check stop loss
            if pos.is_stopped_out(current_price) {
                return Some(ExitReason::StopLoss);
            }

            // Check take profit
            if pos.is_target_hit(current_price) {
                return Some(ExitReason::TakeProfit);
            }

            // Check timeout
            if pos.holding_duration().num_seconds() > self.max_holding_time {
                return Some(ExitReason::Timeout);
            }
        }

        None
    }

    /// Get current position
    pub fn position(&self) -> Option<&Position> {
        self.current_position.as_ref()
    }

    /// Check if flat
    pub fn is_flat(&self) -> bool {
        self.current_position.is_none()
    }

    /// Get daily P&L
    pub fn daily_pnl(&self) -> f64 {
        self.daily_pnl
    }

    /// Reset daily stats
    pub fn reset_daily(&mut self) {
        self.daily_pnl = 0.0;
    }

    /// Get closed trades
    pub fn closed_trades(&self) -> &[ClosedTrade] {
        &self.closed_trades
    }

    /// Get number of trades today
    pub fn trades_today(&self) -> usize {
        let today = Utc::now().date_naive();
        self.closed_trades
            .iter()
            .filter(|t| t.exit_time.date_naive() == today)
            .count()
    }
}

impl Default for PositionManager {
    fn default() -> Self {
        Self::new(1.0, 300, 1000.0) // 1 BTC, 5 min, $1000 max loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_pnl() {
        let mut pos = Position::new(
            "BTCUSDT".to_string(),
            PositionSide::Long,
            1.0,
            50000.0,
            Some(49500.0),
            Some(51000.0),
        );

        pos.update_pnl(50500.0);
        assert!((pos.unrealized_pnl - 500.0).abs() < 0.01);

        pos.update_pnl(49000.0);
        assert!((pos.unrealized_pnl - (-1000.0)).abs() < 0.01);
    }

    #[test]
    fn test_stop_loss() {
        let pos = Position::new(
            "BTCUSDT".to_string(),
            PositionSide::Long,
            1.0,
            50000.0,
            Some(49500.0),
            Some(51000.0),
        );

        assert!(!pos.is_stopped_out(49600.0));
        assert!(pos.is_stopped_out(49400.0));
    }

    #[test]
    fn test_position_manager() {
        let mut manager = PositionManager::new(2.0, 300, 1000.0);

        let pos = Position::new(
            "BTCUSDT".to_string(),
            PositionSide::Long,
            1.0,
            50000.0,
            None,
            None,
        );

        assert!(manager.open_position(pos));
        assert!(!manager.is_flat());

        let trade = manager.close_position(50100.0, ExitReason::Signal);
        assert!(trade.is_some());
        assert!(manager.is_flat());
    }
}
