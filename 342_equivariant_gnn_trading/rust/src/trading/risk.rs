//! Risk Management Module

use serde::{Deserialize, Serialize};

/// Risk manager for position sizing and limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskManager {
    /// Maximum position size as fraction of capital
    pub max_position_size: f64,
    /// Maximum total exposure
    pub max_total_exposure: f64,
    /// Stop loss percentage
    pub stop_loss_pct: f64,
    /// Take profit percentage
    pub take_profit_pct: f64,
    /// Maximum drawdown before stopping
    pub max_drawdown: f64,
}

impl Default for RiskManager {
    fn default() -> Self {
        Self {
            max_position_size: 0.1,
            max_total_exposure: 0.5,
            stop_loss_pct: 0.02,
            take_profit_pct: 0.04,
            max_drawdown: 0.15,
        }
    }
}

impl RiskManager {
    /// Create a new risk manager
    pub fn new(max_position: f64, max_exposure: f64) -> Self {
        Self {
            max_position_size: max_position,
            max_total_exposure: max_exposure,
            ..Default::default()
        }
    }

    /// Calculate position size based on volatility
    pub fn calculate_position_size(&self, base_size: f64, volatility: f64) -> f64 {
        let vol_adjusted = base_size / (1.0 + volatility * 10.0);
        vol_adjusted.min(self.max_position_size).max(0.0)
    }

    /// Check if stop loss triggered
    pub fn check_stop_loss(&self, entry_price: f64, current_price: f64, is_long: bool) -> bool {
        let pnl_pct = if is_long {
            (current_price - entry_price) / entry_price
        } else {
            (entry_price - current_price) / entry_price
        };
        pnl_pct < -self.stop_loss_pct
    }

    /// Check if take profit triggered
    pub fn check_take_profit(&self, entry_price: f64, current_price: f64, is_long: bool) -> bool {
        let pnl_pct = if is_long {
            (current_price - entry_price) / entry_price
        } else {
            (entry_price - current_price) / entry_price
        };
        pnl_pct > self.take_profit_pct
    }
}
