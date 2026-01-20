//! Risk management for news-based trading

use serde::{Deserialize, Serialize};

/// Risk management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Maximum portfolio risk per trade (as fraction)
    pub max_risk_per_trade: f64,
    /// Maximum total portfolio risk (as fraction)
    pub max_total_risk: f64,
    /// Maximum drawdown before stopping (as fraction)
    pub max_drawdown: f64,
    /// Stop loss percentage
    pub stop_loss_pct: f64,
    /// Take profit percentage
    pub take_profit_pct: f64,
    /// Enable trailing stop
    pub use_trailing_stop: bool,
    /// Trailing stop distance (as fraction)
    pub trailing_stop_distance: f64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_risk_per_trade: 0.02,   // 2% per trade
            max_total_risk: 0.10,       // 10% total
            max_drawdown: 0.15,         // 15% max drawdown
            stop_loss_pct: 0.05,        // 5% stop loss
            take_profit_pct: 0.10,      // 10% take profit
            use_trailing_stop: true,
            trailing_stop_distance: 0.03, // 3% trailing
        }
    }
}

/// Risk level assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RiskLevel {
    /// Low risk - safe to trade
    Low,
    /// Medium risk - proceed with caution
    Medium,
    /// High risk - reduce exposure
    High,
    /// Critical risk - stop trading
    Critical,
}

impl RiskLevel {
    /// Get position size multiplier for risk level
    pub fn position_multiplier(&self) -> f64 {
        match self {
            RiskLevel::Low => 1.0,
            RiskLevel::Medium => 0.7,
            RiskLevel::High => 0.3,
            RiskLevel::Critical => 0.0,
        }
    }

    /// Check if trading is allowed at this risk level
    pub fn allows_trading(&self) -> bool {
        !matches!(self, RiskLevel::Critical)
    }
}

/// Risk manager for controlling trading exposure
#[derive(Debug)]
pub struct RiskManager {
    config: RiskConfig,
    current_risk: f64,
    peak_equity: f64,
    current_equity: f64,
    trades_today: usize,
    max_trades_per_day: usize,
}

impl RiskManager {
    /// Create a new risk manager
    pub fn new(config: RiskConfig) -> Self {
        Self {
            config,
            current_risk: 0.0,
            peak_equity: 0.0,
            current_equity: 0.0,
            trades_today: 0,
            max_trades_per_day: 20,
        }
    }

    /// Initialize with starting equity
    pub fn init_equity(&mut self, equity: f64) {
        self.current_equity = equity;
        self.peak_equity = equity;
    }

    /// Update current equity and recalculate risk
    pub fn update_equity(&mut self, equity: f64) {
        self.current_equity = equity;
        if equity > self.peak_equity {
            self.peak_equity = equity;
        }
    }

    /// Get current drawdown
    pub fn current_drawdown(&self) -> f64 {
        if self.peak_equity == 0.0 {
            return 0.0;
        }
        (self.peak_equity - self.current_equity) / self.peak_equity
    }

    /// Assess current risk level
    pub fn assess_risk(&self) -> RiskLevel {
        let drawdown = self.current_drawdown();

        if drawdown >= self.config.max_drawdown {
            RiskLevel::Critical
        } else if drawdown >= self.config.max_drawdown * 0.7 {
            RiskLevel::High
        } else if drawdown >= self.config.max_drawdown * 0.4 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        }
    }

    /// Check if a trade is allowed
    pub fn can_trade(&self) -> TradePermission {
        let risk_level = self.assess_risk();

        if !risk_level.allows_trading() {
            return TradePermission::Denied {
                reason: "Critical risk level - max drawdown exceeded".to_string(),
            };
        }

        if self.trades_today >= self.max_trades_per_day {
            return TradePermission::Denied {
                reason: "Maximum daily trades reached".to_string(),
            };
        }

        if self.current_risk >= self.config.max_total_risk {
            return TradePermission::Denied {
                reason: "Maximum total risk reached".to_string(),
            };
        }

        TradePermission::Allowed {
            max_size: self.calculate_max_position_size(risk_level),
        }
    }

    /// Calculate maximum position size given current risk
    fn calculate_max_position_size(&self, risk_level: RiskLevel) -> f64 {
        let base_size = self.config.max_risk_per_trade / self.config.stop_loss_pct;
        let available_risk = self.config.max_total_risk - self.current_risk;
        let risk_adjusted = base_size * risk_level.position_multiplier();

        risk_adjusted.min(available_risk / self.config.stop_loss_pct)
    }

    /// Calculate stop loss price
    pub fn calculate_stop_loss(&self, entry_price: f64, is_long: bool) -> f64 {
        if is_long {
            entry_price * (1.0 - self.config.stop_loss_pct)
        } else {
            entry_price * (1.0 + self.config.stop_loss_pct)
        }
    }

    /// Calculate take profit price
    pub fn calculate_take_profit(&self, entry_price: f64, is_long: bool) -> f64 {
        if is_long {
            entry_price * (1.0 + self.config.take_profit_pct)
        } else {
            entry_price * (1.0 - self.config.take_profit_pct)
        }
    }

    /// Calculate trailing stop price
    pub fn calculate_trailing_stop(
        &self,
        current_price: f64,
        highest_price: f64,
        is_long: bool,
    ) -> Option<f64> {
        if !self.config.use_trailing_stop {
            return None;
        }

        if is_long {
            Some(highest_price * (1.0 - self.config.trailing_stop_distance))
        } else {
            // For shorts, use lowest price
            Some(current_price * (1.0 + self.config.trailing_stop_distance))
        }
    }

    /// Register a new trade
    pub fn register_trade(&mut self, risk_amount: f64) {
        self.current_risk += risk_amount;
        self.trades_today += 1;
    }

    /// Close a trade and release risk
    pub fn close_trade(&mut self, risk_amount: f64) {
        self.current_risk = (self.current_risk - risk_amount).max(0.0);
    }

    /// Reset daily counters
    pub fn reset_daily(&mut self) {
        self.trades_today = 0;
    }

    /// Get current risk configuration
    pub fn config(&self) -> &RiskConfig {
        &self.config
    }
}

/// Result of trade permission check
#[derive(Debug, Clone)]
pub enum TradePermission {
    /// Trade is allowed with maximum size
    Allowed { max_size: f64 },
    /// Trade is denied with reason
    Denied { reason: String },
}

impl TradePermission {
    /// Check if trading is allowed
    pub fn is_allowed(&self) -> bool {
        matches!(self, TradePermission::Allowed { .. })
    }

    /// Get maximum allowed size if permitted
    pub fn max_size(&self) -> Option<f64> {
        match self {
            TradePermission::Allowed { max_size } => Some(*max_size),
            TradePermission::Denied { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_manager_creation() {
        let config = RiskConfig::default();
        let mut manager = RiskManager::new(config);
        manager.init_equity(10000.0);

        assert_eq!(manager.current_drawdown(), 0.0);
        assert_eq!(manager.assess_risk(), RiskLevel::Low);
    }

    #[test]
    fn test_drawdown_calculation() {
        let mut manager = RiskManager::new(RiskConfig::default());
        manager.init_equity(10000.0);
        manager.update_equity(9000.0);

        assert!((manager.current_drawdown() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_critical_risk_level() {
        let config = RiskConfig {
            max_drawdown: 0.15,
            ..Default::default()
        };
        let mut manager = RiskManager::new(config);
        manager.init_equity(10000.0);
        manager.update_equity(8400.0); // 16% drawdown

        assert_eq!(manager.assess_risk(), RiskLevel::Critical);
        assert!(!manager.can_trade().is_allowed());
    }

    #[test]
    fn test_stop_loss_calculation() {
        let manager = RiskManager::new(RiskConfig {
            stop_loss_pct: 0.05,
            ..Default::default()
        });

        let stop_long = manager.calculate_stop_loss(100.0, true);
        assert!((stop_long - 95.0).abs() < 0.001);

        let stop_short = manager.calculate_stop_loss(100.0, false);
        assert!((stop_short - 105.0).abs() < 0.001);
    }

    #[test]
    fn test_take_profit_calculation() {
        let manager = RiskManager::new(RiskConfig {
            take_profit_pct: 0.10,
            ..Default::default()
        });

        let tp_long = manager.calculate_take_profit(100.0, true);
        assert!((tp_long - 110.0).abs() < 0.001);

        let tp_short = manager.calculate_take_profit(100.0, false);
        assert!((tp_short - 90.0).abs() < 0.001);
    }

    #[test]
    fn test_risk_level_multipliers() {
        assert_eq!(RiskLevel::Low.position_multiplier(), 1.0);
        assert_eq!(RiskLevel::Critical.position_multiplier(), 0.0);
    }
}
