//! Risk management
//!
//! This module provides risk management utilities including
//! stop-loss, take-profit, and drawdown limits.

use serde::{Deserialize, Serialize};

use super::position::{Position, PositionState};

/// Risk management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Maximum drawdown limit (fraction)
    pub max_drawdown: f64,
    /// Daily loss limit (fraction)
    pub daily_loss_limit: f64,
    /// Stop-loss (fraction of entry price)
    pub stop_loss: f64,
    /// Take-profit (fraction of entry price)
    pub take_profit: f64,
    /// ATR multiplier for dynamic stop-loss
    pub atr_multiplier: f64,
    /// Trailing stop activation (fraction of profit)
    pub trailing_stop_activation: f64,
    /// Trailing stop distance (fraction)
    pub trailing_stop_distance: f64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_drawdown: 0.15,
            daily_loss_limit: 0.03,
            stop_loss: 0.02,
            take_profit: 0.04,
            atr_multiplier: 2.0,
            trailing_stop_activation: 0.02,
            trailing_stop_distance: 0.01,
        }
    }
}

/// Risk management decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskAction {
    /// Continue trading
    Continue,
    /// Close position (stop-loss)
    StopLoss,
    /// Close position (take-profit)
    TakeProfit,
    /// Close position (trailing stop)
    TrailingStop,
    /// Halt trading (max drawdown)
    HaltDrawdown,
    /// Halt trading (daily loss limit)
    HaltDailyLoss,
}

/// Risk manager
#[derive(Debug, Clone)]
pub struct RiskManager {
    config: RiskConfig,
    /// Peak equity for drawdown calculation
    peak_equity: f64,
    /// Current drawdown
    current_drawdown: f64,
    /// Daily starting equity
    daily_start_equity: f64,
    /// Daily PnL
    daily_pnl: f64,
    /// Trailing stop price (if active)
    trailing_stop_price: Option<f64>,
    /// Whether trading is halted
    halted: bool,
}

impl RiskManager {
    /// Create a new risk manager
    pub fn new(config: RiskConfig, initial_equity: f64) -> Self {
        Self {
            config,
            peak_equity: initial_equity,
            current_drawdown: 0.0,
            daily_start_equity: initial_equity,
            daily_pnl: 0.0,
            trailing_stop_price: None,
            halted: false,
        }
    }

    /// Check if trading is halted
    pub fn is_halted(&self) -> bool {
        self.halted
    }

    /// Get current drawdown
    pub fn drawdown(&self) -> f64 {
        self.current_drawdown
    }

    /// Get daily PnL
    pub fn daily_pnl(&self) -> f64 {
        self.daily_pnl
    }

    /// Update equity and check drawdown limits
    pub fn update_equity(&mut self, current_equity: f64) -> RiskAction {
        // Update peak equity
        if current_equity > self.peak_equity {
            self.peak_equity = current_equity;
        }

        // Calculate drawdown
        self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity;

        // Check max drawdown
        if self.current_drawdown >= self.config.max_drawdown {
            self.halted = true;
            return RiskAction::HaltDrawdown;
        }

        // Calculate daily PnL
        self.daily_pnl = (current_equity - self.daily_start_equity) / self.daily_start_equity;

        // Check daily loss limit
        if self.daily_pnl <= -self.config.daily_loss_limit {
            self.halted = true;
            return RiskAction::HaltDailyLoss;
        }

        RiskAction::Continue
    }

    /// Check position risk
    pub fn check_position(&mut self, position: &Position, current_price: f64, atr: Option<f64>) -> RiskAction {
        if position.state == PositionState::Flat {
            self.trailing_stop_price = None;
            return RiskAction::Continue;
        }

        let entry_price = position.entry_price;
        let return_pct = match position.state {
            PositionState::Long => (current_price - entry_price) / entry_price,
            PositionState::Short => (entry_price - current_price) / entry_price,
            PositionState::Flat => return RiskAction::Continue,
        };

        // Check stop-loss
        let stop_loss = if let Some(atr) = atr {
            (atr * self.config.atr_multiplier) / entry_price
        } else {
            self.config.stop_loss
        };

        if return_pct <= -stop_loss {
            return RiskAction::StopLoss;
        }

        // Check take-profit
        if return_pct >= self.config.take_profit {
            return RiskAction::TakeProfit;
        }

        // Trailing stop logic
        if return_pct >= self.config.trailing_stop_activation {
            let trailing_price = match position.state {
                PositionState::Long => {
                    current_price * (1.0 - self.config.trailing_stop_distance)
                }
                PositionState::Short => {
                    current_price * (1.0 + self.config.trailing_stop_distance)
                }
                _ => current_price,
            };

            // Update trailing stop
            self.trailing_stop_price = Some(
                self.trailing_stop_price
                    .map(|p| match position.state {
                        PositionState::Long => p.max(trailing_price),
                        PositionState::Short => p.min(trailing_price),
                        _ => trailing_price,
                    })
                    .unwrap_or(trailing_price),
            );

            // Check if trailing stop hit
            if let Some(stop_price) = self.trailing_stop_price {
                let hit = match position.state {
                    PositionState::Long => current_price <= stop_price,
                    PositionState::Short => current_price >= stop_price,
                    _ => false,
                };
                if hit {
                    return RiskAction::TrailingStop;
                }
            }
        }

        RiskAction::Continue
    }

    /// Reset for new day
    pub fn new_day(&mut self, current_equity: f64) {
        self.daily_start_equity = current_equity;
        self.daily_pnl = 0.0;
        self.halted = false;
    }

    /// Reset trailing stop
    pub fn reset_trailing_stop(&mut self) {
        self.trailing_stop_price = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_manager_drawdown() {
        let config = RiskConfig::default();
        let mut manager = RiskManager::new(config, 100000.0);

        // Normal update
        let action = manager.update_equity(98000.0);
        assert_eq!(action, RiskAction::Continue);

        // Hit drawdown limit
        let action = manager.update_equity(84000.0);
        assert_eq!(action, RiskAction::HaltDrawdown);
        assert!(manager.is_halted());
    }

    #[test]
    fn test_stop_loss() {
        let config = RiskConfig {
            stop_loss: 0.02,
            ..Default::default()
        };
        let mut manager = RiskManager::new(config, 100000.0);

        let position = Position {
            state: PositionState::Long,
            size: 1.0,
            entry_price: 50000.0,
            entry_time: 0,
            unrealized_pnl: 0.0,
        };

        // Price dropped 3% -> stop loss
        let action = manager.check_position(&position, 48500.0, None);
        assert_eq!(action, RiskAction::StopLoss);
    }

    #[test]
    fn test_take_profit() {
        let config = RiskConfig {
            take_profit: 0.04,
            ..Default::default()
        };
        let mut manager = RiskManager::new(config, 100000.0);

        let position = Position {
            state: PositionState::Long,
            size: 1.0,
            entry_price: 50000.0,
            entry_time: 0,
            unrealized_pnl: 0.0,
        };

        // Price increased 5% -> take profit
        let action = manager.check_position(&position, 52500.0, None);
        assert_eq!(action, RiskAction::TakeProfit);
    }
}
