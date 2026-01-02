//! Risk Management
//!
//! Position sizing and risk management utilities.

use serde::{Deserialize, Serialize};

/// Risk manager for position sizing and risk control
#[derive(Debug, Clone)]
pub struct RiskManager {
    /// Maximum risk per trade as fraction of capital
    pub max_risk_per_trade: f64,

    /// Maximum total exposure as fraction of capital
    pub max_exposure: f64,

    /// Maximum drawdown before stopping
    pub max_drawdown: f64,

    /// Current drawdown
    pub current_drawdown: f64,

    /// Whether trading is halted due to drawdown
    pub is_halted: bool,
}

impl Default for RiskManager {
    fn default() -> Self {
        Self {
            max_risk_per_trade: 0.02, // 2% per trade
            max_exposure: 0.5,        // 50% of capital
            max_drawdown: 0.20,       // 20% drawdown limit
            current_drawdown: 0.0,
            is_halted: false,
        }
    }
}

impl RiskManager {
    /// Create with custom parameters
    pub fn new(max_risk_per_trade: f64, max_exposure: f64, max_drawdown: f64) -> Self {
        Self {
            max_risk_per_trade: max_risk_per_trade.clamp(0.001, 0.1),
            max_exposure: max_exposure.clamp(0.1, 1.0),
            max_drawdown: max_drawdown.clamp(0.05, 0.5),
            current_drawdown: 0.0,
            is_halted: false,
        }
    }

    /// Calculate position size based on risk
    ///
    /// Uses fixed fractional position sizing:
    /// Position Size = (Capital * Risk%) / (Entry - StopLoss)
    pub fn calculate_position_size(
        &self,
        capital: f64,
        entry_price: f64,
        stop_loss: f64,
        confidence: f64,
    ) -> f64 {
        if self.is_halted {
            return 0.0;
        }

        let risk_amount = capital * self.max_risk_per_trade;
        let risk_per_unit = (entry_price - stop_loss).abs();

        if risk_per_unit == 0.0 {
            return 0.0;
        }

        // Base position size from risk
        let base_size = risk_amount / risk_per_unit;

        // Adjust by confidence
        let adjusted_size = base_size * confidence;

        // Limit by max exposure
        let max_size = (capital * self.max_exposure) / entry_price;

        adjusted_size.min(max_size)
    }

    /// Calculate position size using Kelly Criterion
    ///
    /// Kelly % = (bp - q) / b
    /// where:
    /// - b = odds ratio (reward / risk)
    /// - p = probability of winning
    /// - q = probability of losing (1 - p)
    pub fn kelly_position_size(
        &self,
        capital: f64,
        win_rate: f64,
        avg_win: f64,
        avg_loss: f64,
    ) -> f64 {
        if avg_loss == 0.0 || self.is_halted {
            return 0.0;
        }

        let b = avg_win / avg_loss;
        let p = win_rate;
        let q = 1.0 - p;

        let kelly = (b * p - q) / b;

        // Use half Kelly for safety
        let half_kelly = kelly * 0.5;

        if half_kelly <= 0.0 {
            return 0.0;
        }

        (capital * half_kelly).min(capital * self.max_exposure)
    }

    /// Update drawdown and check if trading should halt
    pub fn update_drawdown(&mut self, current_equity: f64, peak_equity: f64) {
        if peak_equity > 0.0 {
            self.current_drawdown = (peak_equity - current_equity) / peak_equity;

            if self.current_drawdown >= self.max_drawdown {
                self.is_halted = true;
            }
        }
    }

    /// Reset halt status (e.g., after drawdown recovery)
    pub fn reset_halt(&mut self) {
        self.is_halted = false;
        self.current_drawdown = 0.0;
    }

    /// Check if a trade is allowed given current exposure
    pub fn can_trade(&self, current_exposure: f64) -> bool {
        !self.is_halted && current_exposure < self.max_exposure
    }

    /// Calculate Value at Risk (VaR) using historical method
    pub fn calculate_var(returns: &[f64], confidence_level: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mut sorted: Vec<f64> = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((1.0 - confidence_level) * returns.len() as f64).floor() as usize;
        sorted.get(index).copied().unwrap_or(0.0).abs()
    }

    /// Calculate Expected Shortfall (CVaR)
    pub fn calculate_cvar(returns: &[f64], confidence_level: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mut sorted: Vec<f64> = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let var_index = ((1.0 - confidence_level) * returns.len() as f64).floor() as usize;

        if var_index == 0 {
            return sorted[0].abs();
        }

        let tail: Vec<f64> = sorted[..var_index].to_vec();
        tail.iter().sum::<f64>().abs() / tail.len() as f64
    }

    /// Calculate maximum position size for given VaR limit
    pub fn position_size_for_var(
        &self,
        capital: f64,
        price: f64,
        expected_volatility: f64,
        max_var_pct: f64,
        confidence_level: f64,
    ) -> f64 {
        if expected_volatility == 0.0 || self.is_halted {
            return 0.0;
        }

        // Z-score for confidence level (approximation)
        let z = match confidence_level {
            c if c >= 0.99 => 2.326,
            c if c >= 0.95 => 1.645,
            c if c >= 0.90 => 1.282,
            _ => 1.0,
        };

        // VaR = Position * Price * Volatility * Z
        // Max Position = (Capital * MaxVaR%) / (Price * Volatility * Z)
        let max_position = (capital * max_var_pct) / (price * expected_volatility * z);

        max_position.min((capital * self.max_exposure) / price)
    }
}

/// Risk metrics for a portfolio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub var_95: f64,
    pub var_99: f64,
    pub cvar_95: f64,
    pub cvar_99: f64,
    pub max_drawdown: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
}

impl RiskMetrics {
    /// Calculate risk metrics from returns
    pub fn from_returns(returns: &[f64]) -> Self {
        let n = returns.len() as f64;
        if n == 0.0 {
            return Self::default();
        }

        // Mean and volatility
        let mean: f64 = returns.iter().sum::<f64>() / n;
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let volatility = variance.sqrt();

        // Downside deviation (for Sortino)
        let downside_variance: f64 = returns
            .iter()
            .filter(|&&r| r < 0.0)
            .map(|r| r.powi(2))
            .sum::<f64>() / n;
        let downside_dev = downside_variance.sqrt();

        // Sharpe and Sortino (assuming 0 risk-free rate)
        let sharpe_ratio = if volatility > 0.0 {
            mean / volatility * (252.0_f64).sqrt()
        } else {
            0.0
        };

        let sortino_ratio = if downside_dev > 0.0 {
            mean / downside_dev * (252.0_f64).sqrt()
        } else {
            0.0
        };

        // Max drawdown from returns
        let mut equity = 1.0;
        let mut peak = 1.0;
        let mut max_dd = 0.0;

        for &r in returns {
            equity *= 1.0 + r;
            if equity > peak {
                peak = equity;
            }
            let dd = (peak - equity) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        Self {
            var_95: RiskManager::calculate_var(returns, 0.95),
            var_99: RiskManager::calculate_var(returns, 0.99),
            cvar_95: RiskManager::calculate_cvar(returns, 0.95),
            cvar_99: RiskManager::calculate_cvar(returns, 0.99),
            max_drawdown: max_dd,
            volatility: volatility * (252.0_f64).sqrt(), // Annualized
            sharpe_ratio,
            sortino_ratio,
        }
    }
}

impl Default for RiskMetrics {
    fn default() -> Self {
        Self {
            var_95: 0.0,
            var_99: 0.0,
            cvar_95: 0.0,
            cvar_99: 0.0,
            max_drawdown: 0.0,
            volatility: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_sizing() {
        let rm = RiskManager::default();

        // Risk 2% on a trade where stop is 5% away
        let size = rm.calculate_position_size(10000.0, 100.0, 95.0, 1.0);

        // Risk amount = 10000 * 0.02 = 200
        // Risk per unit = 100 - 95 = 5
        // Position size = 200 / 5 = 40 units
        assert!((size - 40.0).abs() < 0.01);
    }

    #[test]
    fn test_kelly_position() {
        let rm = RiskManager::default();

        // 60% win rate, 2:1 reward-risk
        let size = rm.kelly_position_size(10000.0, 0.6, 2.0, 1.0);

        // Kelly = (2 * 0.6 - 0.4) / 2 = 0.4
        // Half Kelly = 0.2
        // Position = 10000 * 0.2 = 2000
        assert!(size > 0.0);
        assert!(size <= 10000.0 * rm.max_exposure);
    }

    #[test]
    fn test_drawdown_halt() {
        let mut rm = RiskManager::default();
        rm.max_drawdown = 0.1;

        rm.update_drawdown(9500.0, 10000.0); // 5% DD
        assert!(!rm.is_halted);

        rm.update_drawdown(8900.0, 10000.0); // 11% DD
        assert!(rm.is_halted);

        rm.reset_halt();
        assert!(!rm.is_halted);
    }

    #[test]
    fn test_var_calculation() {
        let returns = vec![-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06];

        let var_95 = RiskManager::calculate_var(&returns, 0.95);
        assert!(var_95 > 0.0);

        let cvar_95 = RiskManager::calculate_cvar(&returns, 0.95);
        assert!(cvar_95 >= var_95); // CVaR should be >= VaR
    }

    #[test]
    fn test_risk_metrics() {
        let returns: Vec<f64> = (0..100)
            .map(|i| (i as f64 * 0.1).sin() * 0.02)
            .collect();

        let metrics = RiskMetrics::from_returns(&returns);

        assert!(metrics.volatility >= 0.0);
        assert!(metrics.max_drawdown >= 0.0);
    }
}
