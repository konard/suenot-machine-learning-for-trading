//! Risk Management Module

use super::signal::{SignalType, TradingSignal};

/// Risk management configuration
#[derive(Debug, Clone)]
pub struct RiskConfig {
    /// Maximum position size as fraction of capital
    pub max_position_size: f64,
    /// Maximum daily loss as fraction of capital
    pub max_daily_loss: f64,
    /// Per-trade stop loss percentage
    pub stop_loss_pct: f64,
    /// Per-trade take profit percentage
    pub take_profit_pct: f64,
    /// Maximum drawdown before reducing exposure
    pub max_drawdown_pct: f64,
    /// Maximum number of concurrent positions
    pub max_positions: usize,
    /// Maximum exposure per asset
    pub max_asset_exposure: f64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_position_size: 0.1,     // 10% max per trade
            max_daily_loss: 0.02,       // 2% max daily loss
            stop_loss_pct: 0.02,        // 2% stop loss
            take_profit_pct: 0.04,      // 4% take profit (2:1 R/R)
            max_drawdown_pct: 0.10,     // 10% max drawdown
            max_positions: 5,
            max_asset_exposure: 0.20,   // 20% max per asset
        }
    }
}

impl RiskConfig {
    /// Create a conservative risk configuration
    pub fn conservative() -> Self {
        Self {
            max_position_size: 0.05,
            max_daily_loss: 0.01,
            stop_loss_pct: 0.01,
            take_profit_pct: 0.02,
            max_drawdown_pct: 0.05,
            max_positions: 3,
            max_asset_exposure: 0.10,
        }
    }

    /// Create an aggressive risk configuration
    pub fn aggressive() -> Self {
        Self {
            max_position_size: 0.20,
            max_daily_loss: 0.05,
            stop_loss_pct: 0.03,
            take_profit_pct: 0.06,
            max_drawdown_pct: 0.20,
            max_positions: 10,
            max_asset_exposure: 0.30,
        }
    }
}

/// Current portfolio state for risk checking
#[derive(Debug, Clone, Default)]
pub struct PortfolioState {
    /// Current capital
    pub capital: f64,
    /// Daily P&L
    pub daily_pnl: f64,
    /// Current drawdown
    pub current_drawdown: f64,
    /// Number of open positions
    pub open_positions: usize,
    /// Current exposure per asset
    pub asset_exposures: std::collections::HashMap<String, f64>,
}

impl PortfolioState {
    /// Create a new portfolio state
    pub fn new(capital: f64) -> Self {
        Self {
            capital,
            daily_pnl: 0.0,
            current_drawdown: 0.0,
            open_positions: 0,
            asset_exposures: std::collections::HashMap::new(),
        }
    }

    /// Update daily P&L
    pub fn update_pnl(&mut self, pnl: f64) {
        self.daily_pnl += pnl;
    }

    /// Reset daily P&L (called at day boundary)
    pub fn reset_daily(&mut self) {
        self.daily_pnl = 0.0;
    }

    /// Update drawdown
    pub fn update_drawdown(&mut self, peak_capital: f64) {
        self.current_drawdown = (peak_capital - self.capital) / peak_capital;
    }
}

/// Result of signal validation
#[derive(Debug, Clone)]
pub enum ValidatedSignal {
    /// Signal approved (possibly with adjustments)
    Approved(TradingSignal),
    /// Signal blocked with reason
    Blocked(String),
    /// Signal reduced with reason
    Reduced(TradingSignal, String),
}

impl ValidatedSignal {
    /// Check if signal was approved
    pub fn is_approved(&self) -> bool {
        matches!(self, ValidatedSignal::Approved(_) | ValidatedSignal::Reduced(_, _))
    }

    /// Get the signal if approved
    pub fn get_signal(&self) -> Option<&TradingSignal> {
        match self {
            ValidatedSignal::Approved(s) | ValidatedSignal::Reduced(s, _) => Some(s),
            ValidatedSignal::Blocked(_) => None,
        }
    }
}

/// Risk manager
#[derive(Debug)]
pub struct RiskManager {
    /// Configuration
    pub config: RiskConfig,
}

impl Default for RiskManager {
    fn default() -> Self {
        Self {
            config: RiskConfig::default(),
        }
    }
}

impl RiskManager {
    /// Create a new risk manager
    pub fn new(config: RiskConfig) -> Self {
        Self { config }
    }

    /// Validate a trading signal against risk rules
    pub fn validate_signal(
        &self,
        signal: &TradingSignal,
        state: &PortfolioState,
        asset: Option<&str>,
    ) -> ValidatedSignal {
        // Skip validation for neutral signals
        if signal.signal_type == SignalType::Neutral {
            return ValidatedSignal::Approved(signal.clone());
        }

        let mut adjusted = signal.clone();
        let mut reasons = Vec::new();

        // Check daily loss limit
        if state.daily_pnl < -self.config.max_daily_loss * state.capital {
            return ValidatedSignal::Blocked("Daily loss limit reached".to_string());
        }

        // Check maximum drawdown
        if state.current_drawdown > self.config.max_drawdown_pct {
            return ValidatedSignal::Blocked(format!(
                "Maximum drawdown exceeded: {:.1}%",
                state.current_drawdown * 100.0
            ));
        }

        // Check position limit
        if state.open_positions >= self.config.max_positions {
            return ValidatedSignal::Blocked(format!(
                "Maximum positions reached: {}",
                self.config.max_positions
            ));
        }

        // Check asset exposure
        if let Some(asset_name) = asset {
            if let Some(&exposure) = state.asset_exposures.get(asset_name) {
                if exposure >= self.config.max_asset_exposure {
                    return ValidatedSignal::Blocked(format!(
                        "Maximum exposure for {} reached: {:.1}%",
                        asset_name,
                        exposure * 100.0
                    ));
                }
            }
        }

        // Reduce exposure if in significant drawdown
        if state.current_drawdown > self.config.max_drawdown_pct * 0.5 {
            let reduction = 1.0 - (state.current_drawdown / self.config.max_drawdown_pct);
            adjusted.position_size *= reduction;
            reasons.push(format!(
                "Position reduced by {:.0}% due to drawdown",
                (1.0 - reduction) * 100.0
            ));
        }

        // Reduce if approaching daily loss limit
        let daily_loss_ratio = -state.daily_pnl / (self.config.max_daily_loss * state.capital);
        if daily_loss_ratio > 0.5 {
            let reduction = 1.0 - (daily_loss_ratio - 0.5);
            adjusted.position_size *= reduction.max(0.25);
            reasons.push(format!(
                "Position reduced due to daily loss: {:.1}%",
                state.daily_pnl / state.capital * 100.0
            ));
        }

        // Apply maximum position size
        if adjusted.position_size > self.config.max_position_size {
            adjusted.position_size = self.config.max_position_size;
            reasons.push("Position capped at maximum size".to_string());
        }

        // Return result
        if reasons.is_empty() {
            ValidatedSignal::Approved(adjusted)
        } else {
            ValidatedSignal::Reduced(adjusted, reasons.join("; "))
        }
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

    /// Calculate position value from size
    pub fn position_value(&self, capital: f64, position_size: f64) -> f64 {
        capital * position_size.min(self.config.max_position_size)
    }

    /// Calculate risk/reward ratio
    pub fn risk_reward_ratio(&self) -> f64 {
        self.config.take_profit_pct / self.config.stop_loss_pct
    }

    /// Check if trade meets minimum R/R requirement
    pub fn meets_rr_requirement(&self, expected_return: f64, min_rr: f64) -> bool {
        expected_return / self.config.stop_loss_pct >= min_rr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_config_default() {
        let config = RiskConfig::default();
        assert!(config.max_position_size > 0.0);
        assert!(config.max_position_size <= 1.0);
        assert!(config.stop_loss_pct > 0.0);
    }

    #[test]
    fn test_portfolio_state() {
        let mut state = PortfolioState::new(100_000.0);
        state.update_pnl(-500.0);
        assert_eq!(state.daily_pnl, -500.0);

        state.reset_daily();
        assert_eq!(state.daily_pnl, 0.0);
    }

    #[test]
    fn test_validate_signal_approved() {
        let manager = RiskManager::default();
        let state = PortfolioState::new(100_000.0);
        let signal = TradingSignal::new(SignalType::Long, 0.7, 0.05, 0.02);

        let result = manager.validate_signal(&signal, &state, None);
        assert!(result.is_approved());
    }

    #[test]
    fn test_validate_signal_blocked_daily_loss() {
        let manager = RiskManager::default();
        let mut state = PortfolioState::new(100_000.0);
        state.daily_pnl = -3000.0; // Exceeds 2% daily loss limit

        let signal = TradingSignal::new(SignalType::Long, 0.7, 0.05, 0.02);
        let result = manager.validate_signal(&signal, &state, None);

        assert!(!result.is_approved());
    }

    #[test]
    fn test_stop_loss_calculation() {
        let manager = RiskManager::default();

        let long_sl = manager.calculate_stop_loss(100.0, true);
        assert!(long_sl < 100.0);

        let short_sl = manager.calculate_stop_loss(100.0, false);
        assert!(short_sl > 100.0);
    }

    #[test]
    fn test_take_profit_calculation() {
        let manager = RiskManager::default();

        let long_tp = manager.calculate_take_profit(100.0, true);
        assert!(long_tp > 100.0);

        let short_tp = manager.calculate_take_profit(100.0, false);
        assert!(short_tp < 100.0);
    }

    #[test]
    fn test_risk_reward_ratio() {
        let manager = RiskManager::default();
        let rr = manager.risk_reward_ratio();
        assert!(rr >= 1.0); // Should be at least 1:1
    }
}
