//! Trading strategy implementation using MAML.

use crate::maml::algorithm::MAMLTrainer;
use crate::model::network::TradingModel;
use crate::trading::signals::{TradingSignal, DetailedSignal};

/// MAML-based adaptive trading strategy
pub struct TradingStrategy {
    /// MAML trainer for adaptation
    trainer: MAMLTrainer,
    /// Adaptation window size
    adaptation_window: usize,
    /// Number of adaptation steps
    adaptation_steps: usize,
    /// Threshold for trading signals
    threshold: f64,
    /// Strong threshold for trading signals
    strong_threshold: f64,
    /// Stop loss percentage
    stop_loss: f64,
    /// Take profit percentage
    take_profit: f64,
}

impl TradingStrategy {
    /// Create a new trading strategy
    pub fn new(
        model: TradingModel,
        inner_lr: f64,
        outer_lr: f64,
        inner_steps: usize,
        first_order: bool,
    ) -> Self {
        Self {
            trainer: MAMLTrainer::new(model, inner_lr, outer_lr, inner_steps, first_order),
            adaptation_window: 20,
            adaptation_steps: 5,
            threshold: 0.001,
            strong_threshold: 0.005,
            stop_loss: 0.02,
            take_profit: 0.04,
        }
    }

    /// Set adaptation parameters
    pub fn with_adaptation(mut self, window: usize, steps: usize) -> Self {
        self.adaptation_window = window;
        self.adaptation_steps = steps;
        self
    }

    /// Set signal thresholds
    pub fn with_thresholds(mut self, threshold: f64, strong_threshold: f64) -> Self {
        self.threshold = threshold;
        self.strong_threshold = strong_threshold;
        self
    }

    /// Set risk management parameters
    pub fn with_risk_management(mut self, stop_loss: f64, take_profit: f64) -> Self {
        self.stop_loss = stop_loss;
        self.take_profit = take_profit;
        self
    }

    /// Adapt the model to recent data and generate signal
    pub fn generate_signal(
        &self,
        recent_features: &[Vec<f64>],
        recent_returns: &[f64],
        current_features: &[f64],
        timestamp: i64,
        symbol: &str,
    ) -> DetailedSignal {
        // Ensure we have enough data for adaptation
        let n = recent_features.len().min(recent_returns.len()).min(self.adaptation_window);
        if n == 0 {
            return DetailedSignal::new(
                TradingSignal::Hold,
                0.0,
                0.0,
                timestamp,
                symbol.to_string(),
            );
        }

        // Get adaptation data
        let adapt_features: Vec<Vec<f64>> = recent_features[recent_features.len() - n..].to_vec();
        let adapt_labels: Vec<f64> = recent_returns[recent_returns.len() - n..].to_vec();

        // Adapt model
        let adapted_model = self.trainer.adapt(
            &adapt_features,
            &adapt_labels,
            Some(self.adaptation_steps),
        );

        // Generate prediction
        let prediction = adapted_model.predict(current_features);

        // Create signal
        DetailedSignal::from_prediction(
            prediction,
            self.threshold,
            self.strong_threshold,
            timestamp,
            symbol.to_string(),
        )
    }

    /// Check if stop loss should be triggered
    pub fn check_stop_loss(&self, entry_price: f64, current_price: f64, position: i32) -> bool {
        if position == 0 {
            return false;
        }

        let pnl_pct = if position > 0 {
            (current_price / entry_price) - 1.0
        } else {
            1.0 - (current_price / entry_price)
        };

        pnl_pct < -self.stop_loss
    }

    /// Check if take profit should be triggered
    pub fn check_take_profit(&self, entry_price: f64, current_price: f64, position: i32) -> bool {
        if position == 0 {
            return false;
        }

        let pnl_pct = if position > 0 {
            (current_price / entry_price) - 1.0
        } else {
            1.0 - (current_price / entry_price)
        };

        pnl_pct > self.take_profit
    }

    /// Get a reference to the trainer
    pub fn trainer(&self) -> &MAMLTrainer {
        &self.trainer
    }

    /// Get a mutable reference to the trainer
    pub fn trainer_mut(&mut self) -> &mut MAMLTrainer {
        &mut self.trainer
    }

    /// Get adaptation window
    pub fn adaptation_window(&self) -> usize {
        self.adaptation_window
    }

    /// Get adaptation steps
    pub fn adaptation_steps(&self) -> usize {
        self.adaptation_steps
    }

    /// Get threshold
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Get strong threshold
    pub fn strong_threshold(&self) -> f64 {
        self.strong_threshold
    }

    /// Get stop loss
    pub fn stop_loss(&self) -> f64 {
        self.stop_loss
    }

    /// Get take profit
    pub fn take_profit(&self) -> f64 {
        self.take_profit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_creation() {
        let model = TradingModel::new(11, 64, 1);
        let strategy = TradingStrategy::new(model, 0.01, 0.001, 5, true)
            .with_adaptation(30, 10)
            .with_thresholds(0.002, 0.01)
            .with_risk_management(0.03, 0.06);

        assert_eq!(strategy.adaptation_window(), 30);
        assert_eq!(strategy.adaptation_steps(), 10);
        assert!((strategy.threshold() - 0.002).abs() < 1e-10);
        assert!((strategy.stop_loss() - 0.03).abs() < 1e-10);
    }

    #[test]
    fn test_stop_loss_check() {
        let model = TradingModel::new(11, 64, 1);
        let strategy = TradingStrategy::new(model, 0.01, 0.001, 5, true)
            .with_risk_management(0.02, 0.04);

        // Long position with loss
        assert!(strategy.check_stop_loss(100.0, 97.0, 1)); // -3% loss
        assert!(!strategy.check_stop_loss(100.0, 99.0, 1)); // -1% loss

        // Short position with loss
        assert!(strategy.check_stop_loss(100.0, 103.0, -1)); // -3% loss
        assert!(!strategy.check_stop_loss(100.0, 101.0, -1)); // -1% loss

        // No position
        assert!(!strategy.check_stop_loss(100.0, 50.0, 0));
    }

    #[test]
    fn test_take_profit_check() {
        let model = TradingModel::new(11, 64, 1);
        let strategy = TradingStrategy::new(model, 0.01, 0.001, 5, true)
            .with_risk_management(0.02, 0.04);

        // Long position with profit
        assert!(strategy.check_take_profit(100.0, 105.0, 1)); // +5% profit
        assert!(!strategy.check_take_profit(100.0, 102.0, 1)); // +2% profit

        // Short position with profit
        assert!(strategy.check_take_profit(100.0, 95.0, -1)); // +5% profit
        assert!(!strategy.check_take_profit(100.0, 98.0, -1)); // +2% profit
    }

    #[test]
    fn test_signal_generation() {
        let model = TradingModel::new(4, 8, 1);
        let strategy = TradingStrategy::new(model, 0.01, 0.001, 3, true);

        let features = vec![vec![0.1, 0.2, 0.3, 0.4]; 20];
        let returns = vec![0.01; 20];
        let current = vec![0.1, 0.2, 0.3, 0.4];

        let signal = strategy.generate_signal(&features, &returns, &current, 1700000000, "BTCUSDT");

        assert!(!signal.symbol.is_empty());
        assert!(signal.prediction.is_finite());
    }
}
