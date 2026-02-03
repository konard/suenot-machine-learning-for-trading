//! QuantNet transfer trading strategy.
//!
//! Combines QuantNet model predictions with signal generation
//! and position management for multi-asset trading.

use crate::model::network::{QuantNetModel, QuantNetPredictions};
use crate::trading::signals::{SignalGenerator, TradingSignal};

/// Position state for the trading strategy.
#[derive(Debug, Clone)]
pub struct Position {
    pub asset_name: String,
    pub signal: TradingSignal,
    pub size: f64,
    pub entry_price: f64,
}

/// QuantNet transfer trading strategy.
///
/// Uses the QuantNet model for prediction and the signal generator
/// to determine trading actions across multiple assets.
pub struct TradingStrategy {
    model: QuantNetModel,
    signal_generator: SignalGenerator,
    positions: Vec<Option<Position>>,
    transaction_cost: f64,
}

impl TradingStrategy {
    /// Create a new strategy.
    ///
    /// # Arguments
    /// * `model` - Trained QuantNet model
    /// * `signal_threshold` - Min signal strength to trade
    /// * `confidence_threshold` - Min confidence to allow trading
    /// * `transaction_cost` - Cost per trade as fraction
    pub fn new(
        model: QuantNetModel,
        signal_threshold: f64,
        confidence_threshold: f64,
        transaction_cost: f64,
    ) -> Self {
        let num_assets = model.num_assets();
        Self {
            model,
            signal_generator: SignalGenerator::new(signal_threshold, confidence_threshold),
            positions: vec![None; num_assets],
            transaction_cost,
        }
    }

    /// Process a new data point and return trading signals for all assets.
    pub fn on_data(&mut self, features: &[f64], current_prices: &[f64]) -> Vec<(String, TradingSignal)> {
        let predictions = self.model.forward(features);
        let signals = self.signal_generator.generate_all(&predictions);

        // Update position tracking
        for (i, (name, signal)) in signals.iter().enumerate() {
            let price = current_prices.get(i).copied().unwrap_or(0.0);

            match (&self.positions[i], signal) {
                (None, TradingSignal::Buy | TradingSignal::Sell) => {
                    self.positions[i] = Some(Position {
                        asset_name: name.clone(),
                        signal: *signal,
                        size: 1.0,
                        entry_price: price,
                    });
                }
                (Some(pos), new_signal) if pos.signal != *new_signal => {
                    self.positions[i] = if *new_signal == TradingSignal::Hold {
                        None
                    } else {
                        Some(Position {
                            asset_name: name.clone(),
                            signal: *new_signal,
                            size: 1.0,
                            entry_price: price,
                        })
                    };
                }
                _ => {}
            }
        }

        signals
    }

    /// Get QuantNet predictions for given features.
    pub fn predict(&self, features: &[f64]) -> QuantNetPredictions {
        self.model.forward(features)
    }

    /// Get current position for an asset.
    pub fn current_position(&self, asset_index: usize) -> Option<&Position> {
        self.positions.get(asset_index).and_then(|p| p.as_ref())
    }

    /// Get all current positions.
    pub fn all_positions(&self) -> &[Option<Position>] {
        &self.positions
    }

    /// Get transaction cost.
    pub fn transaction_cost(&self) -> f64 {
        self.transaction_cost
    }

    /// Reset the strategy state.
    pub fn reset(&mut self) {
        for pos in &mut self.positions {
            *pos = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_creation() {
        let model = QuantNetModel::new(5, 16, 8, &["BTCUSDT", "ETHUSDT"]);
        let strategy = TradingStrategy::new(model, 0.1, 0.5, 0.001);
        assert!(strategy.current_position(0).is_none());
        assert!(strategy.current_position(1).is_none());
    }

    #[test]
    fn test_strategy_prediction() {
        let model = QuantNetModel::new(5, 16, 8, &["BTCUSDT"]);
        let strategy = TradingStrategy::new(model, 0.1, 0.5, 0.001);
        let features = vec![0.1, -0.2, 0.3, 0.0, 0.5];
        let preds = strategy.predict(&features);
        assert_eq!(preds.asset_outputs.len(), 1);
    }

    #[test]
    fn test_strategy_reset() {
        let model = QuantNetModel::new(5, 16, 8, &["BTCUSDT"]);
        let mut strategy = TradingStrategy::new(model, 0.1, 0.5, 0.001);
        let features = vec![0.1, -0.2, 0.3, 0.0, 0.5];
        strategy.on_data(&features, &[50000.0]);
        strategy.reset();
        assert!(strategy.current_position(0).is_none());
    }
}
