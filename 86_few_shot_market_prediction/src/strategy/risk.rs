//! Risk management for few-shot trading strategies.
//!
//! This module provides risk management utilities that integrate with
//! few-shot learning predictions to manage position sizing and exposure.

use crate::strategy::regime::MarketRegime;
use crate::strategy::signal::{SignalType, TradingSignal};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for risk management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Maximum position size as fraction of portfolio (0-1)
    pub max_position_size: f64,
    /// Maximum total exposure across all positions (0-1)
    pub max_total_exposure: f64,
    /// Maximum loss per trade as fraction of portfolio
    pub max_loss_per_trade: f64,
    /// Daily loss limit as fraction of portfolio
    pub daily_loss_limit: f64,
    /// Minimum confidence required to open position
    pub min_confidence: f64,
    /// Enable regime-based risk adjustment
    pub use_regime_adjustment: bool,
    /// Maximum correlation between positions
    pub max_correlation: f64,
    /// Minimum time between trades (seconds)
    pub min_trade_interval: u64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_position_size: 0.1,        // 10% max per position
            max_total_exposure: 0.5,       // 50% max total exposure
            max_loss_per_trade: 0.02,      // 2% max loss per trade
            daily_loss_limit: 0.05,        // 5% daily loss limit
            min_confidence: 0.6,           // 60% min confidence
            use_regime_adjustment: true,
            max_correlation: 0.7,
            min_trade_interval: 60,
        }
    }
}

impl RiskConfig {
    /// Create a conservative risk configuration.
    pub fn conservative() -> Self {
        Self {
            max_position_size: 0.05,
            max_total_exposure: 0.25,
            max_loss_per_trade: 0.01,
            daily_loss_limit: 0.02,
            min_confidence: 0.75,
            use_regime_adjustment: true,
            max_correlation: 0.5,
            min_trade_interval: 300,
        }
    }

    /// Create an aggressive risk configuration.
    pub fn aggressive() -> Self {
        Self {
            max_position_size: 0.2,
            max_total_exposure: 0.8,
            max_loss_per_trade: 0.05,
            daily_loss_limit: 0.1,
            min_confidence: 0.5,
            use_regime_adjustment: true,
            max_correlation: 0.8,
            min_trade_interval: 30,
        }
    }
}

/// Current position in an asset.
#[derive(Debug, Clone)]
pub struct Position {
    /// Asset symbol
    pub symbol: String,
    /// Position size (negative for short)
    pub size: f64,
    /// Entry price
    pub entry_price: f64,
    /// Current price
    pub current_price: f64,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
    /// Stop loss price
    pub stop_loss: Option<f64>,
    /// Take profit price
    pub take_profit: Option<f64>,
    /// Timestamp of position open
    pub opened_at: u64,
}

impl Position {
    /// Calculate unrealized PnL percentage.
    pub fn pnl_percent(&self) -> f64 {
        if self.entry_price == 0.0 {
            return 0.0;
        }
        (self.current_price - self.entry_price) / self.entry_price * self.size.signum() * 100.0
    }

    /// Check if stop loss is hit.
    pub fn is_stop_hit(&self) -> bool {
        if let Some(sl) = self.stop_loss {
            if self.size > 0.0 {
                self.current_price <= sl
            } else {
                self.current_price >= sl
            }
        } else {
            false
        }
    }

    /// Check if take profit is hit.
    pub fn is_tp_hit(&self) -> bool {
        if let Some(tp) = self.take_profit {
            if self.size > 0.0 {
                self.current_price >= tp
            } else {
                self.current_price <= tp
            }
        } else {
            false
        }
    }
}

/// Risk manager for few-shot trading strategies.
pub struct RiskManager {
    /// Risk configuration
    config: RiskConfig,
    /// Current positions
    positions: HashMap<String, Position>,
    /// Daily realized PnL
    daily_pnl: f64,
    /// Portfolio value
    portfolio_value: f64,
    /// Trade timestamps per symbol
    last_trades: HashMap<String, u64>,
}

impl RiskManager {
    /// Create a new risk manager.
    pub fn new(config: RiskConfig, portfolio_value: f64) -> Self {
        Self {
            config,
            positions: HashMap::new(),
            daily_pnl: 0.0,
            portfolio_value,
            last_trades: HashMap::new(),
        }
    }

    /// Check if a signal passes risk checks.
    pub fn check_signal(&self, signal: &TradingSignal, current_time: u64) -> RiskCheckResult {
        let mut result = RiskCheckResult::new();

        // Check confidence threshold
        if signal.confidence < self.config.min_confidence {
            result.add_rejection("Confidence below threshold");
            return result;
        }

        // Check if signal requires action
        if !signal.signal_type.requires_action() {
            result.approved = true;
            return result;
        }

        // Check trade interval
        if let Some(&last_trade) = self.last_trades.get(&signal.symbol) {
            if current_time.saturating_sub(last_trade) < self.config.min_trade_interval {
                result.add_rejection("Trade interval too short");
                return result;
            }
        }

        // Check daily loss limit
        if self.daily_pnl < -self.config.daily_loss_limit * self.portfolio_value {
            result.add_rejection("Daily loss limit reached");
            return result;
        }

        // Check position-specific risks
        match signal.signal_type {
            SignalType::Long | SignalType::Short => {
                // Check if we already have opposite position
                if let Some(pos) = self.positions.get(&signal.symbol) {
                    let is_opposite = (signal.signal_type == SignalType::Long && pos.size < 0.0)
                        || (signal.signal_type == SignalType::Short && pos.size > 0.0);
                    if is_opposite {
                        result.add_warning("Reversing existing position");
                    }
                }

                // Check total exposure
                let current_exposure = self.total_exposure();
                if current_exposure >= self.config.max_total_exposure {
                    result.add_rejection("Maximum total exposure reached");
                    return result;
                }

                // Calculate allowed position size
                let max_new_position = (self.config.max_total_exposure - current_exposure)
                    .min(self.config.max_position_size);

                result.max_position_size = max_new_position;
                result.approved = true;
            }
            SignalType::Close | SignalType::ReducePosition => {
                // Always allow closing/reducing positions
                result.approved = true;
            }
            SignalType::Hold => {
                result.approved = true;
            }
        }

        result
    }

    /// Apply risk adjustments to a trading signal.
    pub fn adjust_signal(&self, signal: &mut TradingSignal, regime: Option<MarketRegime>) {
        // Apply regime-based adjustments
        if self.config.use_regime_adjustment {
            if let Some(r) = regime {
                let regime_factor = self.regime_risk_factor(r);
                signal.position_size *= regime_factor;
            }
        }

        // Cap position size
        signal.position_size = signal.position_size.min(self.config.max_position_size);

        // Ensure proper stop loss based on max loss per trade
        if signal.stop_loss.is_none() || signal.stop_loss.unwrap() > self.config.max_loss_per_trade * 100.0 {
            signal.stop_loss = Some(self.config.max_loss_per_trade * 100.0);
        }
    }

    /// Calculate risk factor based on market regime.
    fn regime_risk_factor(&self, regime: MarketRegime) -> f64 {
        match regime {
            MarketRegime::StrongUptrend | MarketRegime::StrongDowntrend => 1.0,
            MarketRegime::WeakUptrend | MarketRegime::WeakDowntrend => 0.8,
            MarketRegime::Sideways => 0.5,
            MarketRegime::Volatile => 0.4,
            MarketRegime::Crash => 0.2,
            MarketRegime::Recovery => 0.7,
        }
    }

    /// Calculate Kelly criterion position size.
    pub fn kelly_position_size(&self, win_rate: f64, win_loss_ratio: f64) -> f64 {
        // Kelly formula: f* = (bp - q) / b
        // where b = win/loss ratio, p = win probability, q = loss probability
        let b = win_loss_ratio;
        let p = win_rate;
        let q = 1.0 - p;

        let kelly = (b * p - q) / b;

        // Use fractional Kelly (25%) for safety
        let fractional_kelly = kelly * 0.25;

        // Clamp to max position size
        fractional_kelly.max(0.0).min(self.config.max_position_size)
    }

    /// Calculate total exposure as fraction of portfolio.
    pub fn total_exposure(&self) -> f64 {
        if self.portfolio_value == 0.0 {
            return 0.0;
        }

        self.positions
            .values()
            .map(|p| p.size.abs() * p.current_price)
            .sum::<f64>()
            / self.portfolio_value
    }

    /// Get current drawdown.
    pub fn current_drawdown(&self) -> f64 {
        // Simplified: just use daily PnL as proxy
        if self.portfolio_value == 0.0 {
            return 0.0;
        }
        (-self.daily_pnl / self.portfolio_value).max(0.0)
    }

    /// Update position with new price.
    pub fn update_position(&mut self, symbol: &str, current_price: f64) {
        if let Some(pos) = self.positions.get_mut(symbol) {
            pos.current_price = current_price;
            let price_change = current_price - pos.entry_price;
            pos.unrealized_pnl = price_change * pos.size;
        }
    }

    /// Open a new position.
    pub fn open_position(&mut self, symbol: &str, size: f64, entry_price: f64, stop_loss: Option<f64>, take_profit: Option<f64>, timestamp: u64) {
        let position = Position {
            symbol: symbol.to_string(),
            size,
            entry_price,
            current_price: entry_price,
            unrealized_pnl: 0.0,
            stop_loss,
            take_profit,
            opened_at: timestamp,
        };

        self.positions.insert(symbol.to_string(), position);
        self.last_trades.insert(symbol.to_string(), timestamp);
    }

    /// Close a position.
    pub fn close_position(&mut self, symbol: &str) -> Option<f64> {
        if let Some(pos) = self.positions.remove(symbol) {
            let realized_pnl = pos.unrealized_pnl;
            self.daily_pnl += realized_pnl;
            Some(realized_pnl)
        } else {
            None
        }
    }

    /// Get position for a symbol.
    pub fn get_position(&self, symbol: &str) -> Option<&Position> {
        self.positions.get(symbol)
    }

    /// Get all positions.
    pub fn all_positions(&self) -> &HashMap<String, Position> {
        &self.positions
    }

    /// Reset daily PnL (call at start of new day).
    pub fn reset_daily(&mut self) {
        self.daily_pnl = 0.0;
    }

    /// Update portfolio value.
    pub fn set_portfolio_value(&mut self, value: f64) {
        self.portfolio_value = value;
    }

    /// Calculate value at risk (simplified historical VaR).
    pub fn value_at_risk(&self, confidence_level: f64) -> f64 {
        let total_exposure = self.total_exposure();

        // Simplified VaR: assume 2% daily volatility for crypto
        let daily_volatility = 0.02;

        // Z-score for confidence level (e.g., 1.645 for 95%, 2.326 for 99%)
        let z_score = match confidence_level {
            c if c >= 0.99 => 2.326,
            c if c >= 0.95 => 1.645,
            c if c >= 0.90 => 1.282,
            _ => 1.0,
        };

        total_exposure * self.portfolio_value * daily_volatility * z_score
    }

    /// Get a summary of current risk metrics.
    pub fn risk_summary(&self) -> RiskSummary {
        RiskSummary {
            total_exposure: self.total_exposure(),
            daily_pnl: self.daily_pnl,
            daily_pnl_percent: if self.portfolio_value > 0.0 {
                self.daily_pnl / self.portfolio_value * 100.0
            } else {
                0.0
            },
            position_count: self.positions.len(),
            value_at_risk_95: self.value_at_risk(0.95),
            available_exposure: self.config.max_total_exposure - self.total_exposure(),
            daily_limit_remaining: self.config.daily_loss_limit + (self.daily_pnl / self.portfolio_value),
        }
    }
}

/// Result of a risk check.
#[derive(Debug, Clone)]
pub struct RiskCheckResult {
    /// Whether the signal is approved
    pub approved: bool,
    /// Maximum allowed position size
    pub max_position_size: f64,
    /// Rejection reasons
    pub rejections: Vec<String>,
    /// Warnings (signal approved but with caveats)
    pub warnings: Vec<String>,
}

impl RiskCheckResult {
    fn new() -> Self {
        Self {
            approved: false,
            max_position_size: 0.0,
            rejections: Vec::new(),
            warnings: Vec::new(),
        }
    }

    fn add_rejection(&mut self, reason: &str) {
        self.approved = false;
        self.rejections.push(reason.to_string());
    }

    fn add_warning(&mut self, warning: &str) {
        self.warnings.push(warning.to_string());
    }

    /// Check if there are any warnings.
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

/// Summary of current risk metrics.
#[derive(Debug, Clone)]
pub struct RiskSummary {
    /// Total exposure as fraction of portfolio
    pub total_exposure: f64,
    /// Daily realized + unrealized PnL
    pub daily_pnl: f64,
    /// Daily PnL as percentage
    pub daily_pnl_percent: f64,
    /// Number of open positions
    pub position_count: usize,
    /// 95% Value at Risk
    pub value_at_risk_95: f64,
    /// Remaining exposure capacity
    pub available_exposure: f64,
    /// Remaining daily loss limit
    pub daily_limit_remaining: f64,
}

impl std::fmt::Display for RiskSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Risk Summary:\n\
             - Exposure: {:.1}%\n\
             - Daily PnL: {:.2}%\n\
             - Positions: {}\n\
             - VaR (95%): {:.2}\n\
             - Available: {:.1}%",
            self.total_exposure * 100.0,
            self.daily_pnl_percent,
            self.position_count,
            self.value_at_risk_95,
            self.available_exposure * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_config() {
        let config = RiskConfig::default();
        assert!((config.max_position_size - 0.1).abs() < 0.001);

        let conservative = RiskConfig::conservative();
        assert!(conservative.max_position_size < config.max_position_size);

        let aggressive = RiskConfig::aggressive();
        assert!(aggressive.max_position_size > config.max_position_size);
    }

    #[test]
    fn test_position_pnl() {
        let pos = Position {
            symbol: "BTCUSDT".to_string(),
            size: 1.0,
            entry_price: 50000.0,
            current_price: 51000.0,
            unrealized_pnl: 1000.0,
            stop_loss: Some(49000.0),
            take_profit: Some(55000.0),
            opened_at: 0,
        };

        let pnl = pos.pnl_percent();
        assert!((pnl - 2.0).abs() < 0.01); // 2% gain
    }

    #[test]
    fn test_kelly_criterion() {
        let config = RiskConfig::default();
        let manager = RiskManager::new(config, 100000.0);

        // 60% win rate, 2:1 win/loss ratio
        let kelly = manager.kelly_position_size(0.6, 2.0);

        // Kelly = (2 * 0.6 - 0.4) / 2 = 0.4
        // Fractional = 0.4 * 0.25 = 0.1
        assert!(kelly > 0.0);
        assert!(kelly <= 0.1); // Should be capped at max position size
    }

    #[test]
    fn test_risk_check() {
        let config = RiskConfig::default();
        let manager = RiskManager::new(config, 100000.0);

        let signal = TradingSignal::new(SignalType::Long, 0.8, "BTCUSDT");
        let result = manager.check_signal(&signal, 0);

        assert!(result.approved);
    }

    #[test]
    fn test_reject_low_confidence() {
        let config = RiskConfig::default();
        let manager = RiskManager::new(config, 100000.0);

        let signal = TradingSignal::new(SignalType::Long, 0.3, "BTCUSDT"); // Low confidence
        let result = manager.check_signal(&signal, 0);

        assert!(!result.approved);
        assert!(!result.rejections.is_empty());
    }
}
