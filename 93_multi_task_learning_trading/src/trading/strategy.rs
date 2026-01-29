//! Multi-task learning trading strategy.
//!
//! Combines multi-task model predictions with signal generation
//! and position management.

use crate::model::network::{MultiTaskModel, TaskPredictions};
use crate::trading::signals::{SignalGenerator, TradingSignal};

/// Position state for the trading strategy.
#[derive(Debug, Clone, Copy)]
pub struct Position {
    pub signal: TradingSignal,
    pub size: f64,
    pub entry_price: f64,
}

/// Multi-task learning trading strategy.
///
/// Uses the MTL model for prediction and the signal generator
/// to determine trading actions.
pub struct TradingStrategy {
    model: MultiTaskModel,
    signal_generator: SignalGenerator,
    position: Option<Position>,
    transaction_cost: f64,
}

impl TradingStrategy {
    /// Create a new strategy.
    ///
    /// # Arguments
    /// * `model` - Trained multi-task model
    /// * `return_threshold` - Min predicted return to trade
    /// * `direction_threshold` - Min direction confidence
    /// * `volatility_cap` - Max volatility to allow trading
    /// * `transaction_cost` - Cost per trade as fraction
    pub fn new(
        model: MultiTaskModel,
        return_threshold: f64,
        direction_threshold: f64,
        volatility_cap: f64,
        transaction_cost: f64,
    ) -> Self {
        Self {
            model,
            signal_generator: SignalGenerator::new(
                return_threshold,
                direction_threshold,
                volatility_cap,
            ),
            position: None,
            transaction_cost,
        }
    }

    /// Process a new data point and return a trading signal.
    pub fn on_data(&mut self, features: &[f64], current_price: f64) -> TradingSignal {
        let predictions = self.model.forward(features);
        let signal = self.signal_generator.generate(&predictions, 0);

        // Update position tracking
        match (&self.position, signal) {
            (None, TradingSignal::Long | TradingSignal::Short) => {
                self.position = Some(Position {
                    signal,
                    size: 1.0,
                    entry_price: current_price,
                });
            }
            (Some(pos), new_signal) if pos.signal != new_signal => {
                self.position = if new_signal == TradingSignal::Neutral {
                    None
                } else {
                    Some(Position {
                        signal: new_signal,
                        size: 1.0,
                        entry_price: current_price,
                    })
                };
            }
            _ => {}
        }

        signal
    }

    /// Get multi-task predictions for given features.
    pub fn predict(&self, features: &[f64]) -> TaskPredictions {
        self.model.forward(features)
    }

    /// Get batch predictions.
    pub fn predict_batch(&self, features: &[Vec<f64>]) -> TaskPredictions {
        self.model.forward_batch(features)
    }

    /// Get current position.
    pub fn current_position(&self) -> Option<&Position> {
        self.position.as_ref()
    }

    /// Get transaction cost.
    pub fn transaction_cost(&self) -> f64 {
        self.transaction_cost
    }

    /// Reset the strategy state.
    pub fn reset(&mut self) {
        self.position = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_creation() {
        let model = MultiTaskModel::new(5, &[16, 8], 4);
        let strategy = TradingStrategy::new(model, 0.001, 0.55, 0.05, 0.001);
        assert!(strategy.current_position().is_none());
    }

    #[test]
    fn test_strategy_prediction() {
        let model = MultiTaskModel::new(5, &[16, 8], 4);
        let strategy = TradingStrategy::new(model, 0.001, 0.55, 0.05, 0.001);
        let features = vec![0.1, -0.2, 0.3, 0.0, 0.5];
        let preds = strategy.predict(&features);
        assert_eq!(preds.return_pred.len(), 1);
    }

    #[test]
    fn test_strategy_reset() {
        let model = MultiTaskModel::new(5, &[16, 8], 4);
        let mut strategy = TradingStrategy::new(model, 0.001, 0.55, 0.05, 0.001);
        let features = vec![0.1, -0.2, 0.3, 0.0, 0.5];
        strategy.on_data(&features, 100.0);
        strategy.reset();
        assert!(strategy.current_position().is_none());
    }
}
