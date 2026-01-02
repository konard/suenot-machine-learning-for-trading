//! Trading signal generation
//!
//! Generate buy/sell signals from GAT predictions.

use crate::gat::GraphAttentionNetwork;
use crate::graph::SparseGraph;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Trading signal
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Signal {
    StrongBuy,
    Buy,
    Hold,
    Sell,
    StrongSell,
}

impl Signal {
    /// Convert from continuous value [-1, 1]
    pub fn from_value(value: f64) -> Self {
        if value > 0.6 {
            Signal::StrongBuy
        } else if value > 0.2 {
            Signal::Buy
        } else if value > -0.2 {
            Signal::Hold
        } else if value > -0.6 {
            Signal::Sell
        } else {
            Signal::StrongSell
        }
    }

    /// Convert to position size multiplier
    pub fn to_position(&self) -> f64 {
        match self {
            Signal::StrongBuy => 1.0,
            Signal::Buy => 0.5,
            Signal::Hold => 0.0,
            Signal::Sell => -0.5,
            Signal::StrongSell => -1.0,
        }
    }
}

/// Signal generator using GAT
#[derive(Debug)]
pub struct SignalGenerator {
    /// Buy threshold
    buy_threshold: f64,
    /// Sell threshold
    sell_threshold: f64,
    /// Minimum signal strength
    min_strength: f64,
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl SignalGenerator {
    /// Create with default thresholds
    pub fn new() -> Self {
        Self {
            buy_threshold: 0.2,
            sell_threshold: -0.2,
            min_strength: 0.1,
        }
    }

    /// Create with custom thresholds
    pub fn with_thresholds(buy: f64, sell: f64, min_strength: f64) -> Self {
        Self {
            buy_threshold: buy,
            sell_threshold: sell,
            min_strength,
        }
    }

    /// Generate signals from GAT output
    pub fn generate(
        &self,
        gat: &GraphAttentionNetwork,
        features: &Array2<f64>,
        graph: &SparseGraph,
    ) -> Vec<Signal> {
        let raw_signals = gat.predict_signals(features, graph);

        raw_signals
            .iter()
            .map(|&s| {
                if s.abs() < self.min_strength {
                    Signal::Hold
                } else {
                    Signal::from_value(s)
                }
            })
            .collect()
    }

    /// Generate raw signal values
    pub fn generate_raw(
        &self,
        gat: &GraphAttentionNetwork,
        features: &Array2<f64>,
        graph: &SparseGraph,
    ) -> Array1<f64> {
        gat.predict_signals(features, graph)
    }

    /// Apply signal smoothing (EMA)
    pub fn smooth_signals(
        &self,
        current: &Array1<f64>,
        previous: &Array1<f64>,
        alpha: f64,
    ) -> Array1<f64> {
        current * alpha + previous * (1.0 - alpha)
    }
}

/// Trading strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingStrategy {
    /// Strategy name
    pub name: String,
    /// Maximum position size per asset (as fraction of portfolio)
    pub max_position: f64,
    /// Rebalance threshold
    pub rebalance_threshold: f64,
    /// Stop loss percentage
    pub stop_loss: f64,
    /// Take profit percentage
    pub take_profit: f64,
    /// Use trailing stop
    pub trailing_stop: bool,
    /// Signal smoothing factor
    pub signal_smoothing: f64,
}

impl Default for TradingStrategy {
    fn default() -> Self {
        Self::new("GAT_Strategy")
    }
}

impl TradingStrategy {
    /// Create new strategy
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            max_position: 0.2,
            rebalance_threshold: 0.05,
            stop_loss: 0.05,
            take_profit: 0.10,
            trailing_stop: true,
            signal_smoothing: 0.3,
        }
    }

    /// Create aggressive strategy
    pub fn aggressive(name: &str) -> Self {
        Self {
            name: name.to_string(),
            max_position: 0.4,
            rebalance_threshold: 0.03,
            stop_loss: 0.08,
            take_profit: 0.15,
            trailing_stop: true,
            signal_smoothing: 0.5,
        }
    }

    /// Create conservative strategy
    pub fn conservative(name: &str) -> Self {
        Self {
            name: name.to_string(),
            max_position: 0.1,
            rebalance_threshold: 0.10,
            stop_loss: 0.03,
            take_profit: 0.05,
            trailing_stop: false,
            signal_smoothing: 0.2,
        }
    }

    /// Compute target positions from signals
    pub fn compute_positions(&self, signals: &[Signal]) -> Vec<f64> {
        signals
            .iter()
            .map(|s| s.to_position() * self.max_position)
            .collect()
    }

    /// Check if rebalance is needed
    pub fn needs_rebalance(&self, current: &[f64], target: &[f64]) -> bool {
        current
            .iter()
            .zip(target.iter())
            .any(|(c, t)| (c - t).abs() > self.rebalance_threshold)
    }

    /// Check stop loss condition
    pub fn check_stop_loss(&self, entry_price: f64, current_price: f64, is_long: bool) -> bool {
        let pnl = if is_long {
            (current_price - entry_price) / entry_price
        } else {
            (entry_price - current_price) / entry_price
        };

        pnl < -self.stop_loss
    }

    /// Check take profit condition
    pub fn check_take_profit(&self, entry_price: f64, current_price: f64, is_long: bool) -> bool {
        let pnl = if is_long {
            (current_price - entry_price) / entry_price
        } else {
            (entry_price - current_price) / entry_price
        };

        pnl > self.take_profit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_from_value() {
        assert_eq!(Signal::from_value(0.8), Signal::StrongBuy);
        assert_eq!(Signal::from_value(0.3), Signal::Buy);
        assert_eq!(Signal::from_value(0.0), Signal::Hold);
        assert_eq!(Signal::from_value(-0.3), Signal::Sell);
        assert_eq!(Signal::from_value(-0.8), Signal::StrongSell);
    }

    #[test]
    fn test_signal_to_position() {
        assert_eq!(Signal::StrongBuy.to_position(), 1.0);
        assert_eq!(Signal::Hold.to_position(), 0.0);
        assert_eq!(Signal::StrongSell.to_position(), -1.0);
    }

    #[test]
    fn test_strategy_positions() {
        let strategy = TradingStrategy::new("Test");
        let signals = vec![Signal::StrongBuy, Signal::Hold, Signal::Sell];
        let positions = strategy.compute_positions(&signals);

        assert_eq!(positions.len(), 3);
        assert!((positions[0] - 0.2).abs() < 0.001);
        assert!((positions[1] - 0.0).abs() < 0.001);
        assert!((positions[2] - (-0.1)).abs() < 0.001);
    }

    #[test]
    fn test_stop_loss() {
        let strategy = TradingStrategy::new("Test");

        // Long position with 6% loss
        assert!(strategy.check_stop_loss(100.0, 94.0, true));

        // Long position with 4% loss (within limit)
        assert!(!strategy.check_stop_loss(100.0, 96.0, true));
    }
}
