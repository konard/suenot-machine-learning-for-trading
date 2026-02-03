//! S4-based trading strategy implementation.

use crate::model::s4::{S4Model, S4Config};
use crate::trading::signals::TradingSignal;
use crate::data::bybit::MarketData;

/// Configuration for the trading strategy.
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Minimum confidence to execute trade.
    pub min_confidence: f64,
    /// Maximum position size (as fraction of capital).
    pub max_position_size: f64,
    /// Lookback window for features.
    pub lookback: usize,
    /// Stop loss percentage.
    pub stop_loss: f64,
    /// Take profit percentage.
    pub take_profit: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            max_position_size: 0.1,
            lookback: 256,
            stop_loss: 0.02,
            take_profit: 0.04,
        }
    }
}

/// S4-based trading strategy.
pub struct TradingStrategy {
    /// S4 model for signal generation.
    model: S4Model,
    /// Strategy configuration.
    config: StrategyConfig,
    /// Current position.
    position: f64,
    /// Entry price.
    entry_price: Option<f64>,
}

impl TradingStrategy {
    /// Create a new trading strategy.
    pub fn new(model_config: S4Config, strategy_config: StrategyConfig) -> Self {
        Self {
            model: S4Model::new(model_config),
            config: strategy_config,
            position: 0.0,
            entry_price: None,
        }
    }

    /// Create with default configurations.
    pub fn with_defaults() -> Self {
        Self::new(S4Config::default(), StrategyConfig::default())
    }

    /// Generate a trading signal from market data.
    pub fn generate_signal(&self, data: &MarketData) -> TradingSignal {
        let features = data.to_features();

        // Use the last `lookback` data points
        let start = if features.len() > self.config.lookback {
            features.len() - self.config.lookback
        } else {
            0
        };

        let input_features = &features[start..];
        self.model.predict(input_features)
    }

    /// Execute the strategy on new data.
    ///
    /// Returns the signal and suggested position change.
    pub fn on_data(&mut self, data: &MarketData) -> (TradingSignal, f64) {
        let signal = self.generate_signal(data);
        let current_price = data.candles.last().map(|c| c.close).unwrap_or(0.0);

        // Check stop loss / take profit
        if let Some(entry) = self.entry_price {
            if self.position > 0.0 {
                let pnl_pct = (current_price - entry) / entry;
                if pnl_pct <= -self.config.stop_loss || pnl_pct >= self.config.take_profit {
                    // Close position
                    let old_position = self.position;
                    self.position = 0.0;
                    self.entry_price = None;
                    return (signal, -old_position);
                }
            } else if self.position < 0.0 {
                let pnl_pct = (entry - current_price) / entry;
                if pnl_pct <= -self.config.stop_loss || pnl_pct >= self.config.take_profit {
                    // Close position
                    let old_position = self.position;
                    self.position = 0.0;
                    self.entry_price = None;
                    return (signal, -old_position);
                }
            }
        }

        // Determine target position based on signal
        let target_position = signal.position_size(
            self.config.max_position_size,
            self.config.min_confidence,
        );

        let position_change = target_position - self.position;

        // Update position
        if position_change.abs() > 0.001 {
            self.position = target_position;
            if target_position.abs() > 0.001 {
                self.entry_price = Some(current_price);
            } else {
                self.entry_price = None;
            }
        }

        (signal, position_change)
    }

    /// Get current position.
    pub fn position(&self) -> f64 {
        self.position
    }

    /// Get entry price.
    pub fn entry_price(&self) -> Option<f64> {
        self.entry_price
    }

    /// Reset the strategy state.
    pub fn reset(&mut self) {
        self.position = 0.0;
        self.entry_price = None;
    }

    /// Get model's hidden state for analysis.
    pub fn get_model_state(&self, data: &MarketData) -> Vec<f64> {
        let features = data.to_features();
        let start = if features.len() > self.config.lookback {
            features.len() - self.config.lookback
        } else {
            0
        };
        self.model.get_state(&features[start..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::generate_synthetic_data;

    #[test]
    fn test_strategy_creation() {
        let strategy = TradingStrategy::with_defaults();
        assert_eq!(strategy.position(), 0.0);
    }

    #[test]
    fn test_signal_generation() {
        let strategy = TradingStrategy::with_defaults();
        let data = generate_synthetic_data("BTCUSDT", "60", 300);
        let signal = strategy.generate_signal(&data);
        assert!(signal.confidence() >= 0.0 && signal.confidence() <= 1.0);
    }

    #[test]
    fn test_strategy_execution() {
        let mut strategy = TradingStrategy::with_defaults();
        let data = generate_synthetic_data("BTCUSDT", "60", 300);
        let (signal, _change) = strategy.on_data(&data);
        println!("Signal: {}", signal);
    }
}
