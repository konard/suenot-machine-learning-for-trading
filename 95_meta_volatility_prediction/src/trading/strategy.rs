//! Volatility-adaptive trading strategy.
//!
//! Adjusts position sizes based on predicted volatility:
//! position_size = target_risk / predicted_volatility

use crate::trading::signals::{generate_signal, TradingSignal};

/// Configuration for the volatility-adaptive strategy.
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Target daily risk as fraction of portfolio (e.g., 0.02 = 2%).
    pub target_daily_risk: f64,
    /// Maximum position size as fraction of portfolio.
    pub max_position_pct: f64,
    /// Minimum position size as fraction of portfolio.
    pub min_position_pct: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            target_daily_risk: 0.02,
            max_position_pct: 0.5,
            min_position_pct: 0.01,
        }
    }
}

/// Volatility-adaptive position sizing strategy.
pub struct VolatilityStrategy {
    config: StrategyConfig,
    historical_vol: f64,
    prev_vol: f64,
    current_position: f64,
}

impl VolatilityStrategy {
    /// Create a new strategy with given config.
    pub fn new(config: StrategyConfig) -> Self {
        Self {
            config,
            historical_vol: 0.02,
            prev_vol: 0.02,
            current_position: 0.0,
        }
    }

    /// Compute position size based on predicted volatility.
    pub fn compute_position_size(&self, predicted_vol: f64) -> f64 {
        if predicted_vol <= 1e-8 {
            return self.config.max_position_pct;
        }

        let raw_size = self.config.target_daily_risk / predicted_vol;
        raw_size.clamp(self.config.min_position_pct, self.config.max_position_pct)
    }

    /// Update strategy state and return (position_size, signal).
    pub fn update(&mut self, predicted_vol: f64) -> (f64, TradingSignal) {
        let vol_change_rate = if self.prev_vol > 1e-10 {
            (predicted_vol - self.prev_vol) / self.prev_vol
        } else {
            0.0
        };

        let signal = generate_signal(predicted_vol, self.historical_vol, vol_change_rate);

        let position_size = match signal {
            TradingSignal::Exit => 0.0,
            _ => self.compute_position_size(predicted_vol),
        };

        // Update state with exponential moving average
        self.historical_vol = 0.95 * self.historical_vol + 0.05 * predicted_vol;
        self.prev_vol = predicted_vol;
        self.current_position = position_size;

        (position_size, signal)
    }

    /// Get current position size.
    pub fn current_position(&self) -> f64 {
        self.current_position
    }

    /// Get historical volatility estimate.
    pub fn historical_vol(&self) -> f64 {
        self.historical_vol
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_size_inverse_vol() {
        let strategy = VolatilityStrategy::new(StrategyConfig::default());
        let low_vol_pos = strategy.compute_position_size(0.01);
        let high_vol_pos = strategy.compute_position_size(0.05);
        assert!(low_vol_pos > high_vol_pos);
    }

    #[test]
    fn test_position_size_bounds() {
        let config = StrategyConfig::default();
        let strategy = VolatilityStrategy::new(config.clone());

        let pos = strategy.compute_position_size(0.001);
        assert!(pos <= config.max_position_pct);

        let pos = strategy.compute_position_size(100.0);
        assert!(pos >= config.min_position_pct);
    }

    #[test]
    fn test_strategy_update() {
        let mut strategy = VolatilityStrategy::new(StrategyConfig::default());
        let (pos, signal) = strategy.update(0.02);
        assert!(pos > 0.0);
        assert_eq!(signal, TradingSignal::Hold);
    }
}
