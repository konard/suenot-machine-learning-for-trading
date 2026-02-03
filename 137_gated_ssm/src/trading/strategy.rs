//! Gated SSM trading strategy implementation.

use crate::model::GatedSSMBlock;
use super::signals::{SignalDirection, TradingSignal};

/// Sigmoid activation function.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Trading strategy using a Gated SSM model.
///
/// Feeds feature vectors through the Gated SSM block and generates
/// trading signals based on the model's output score.
pub struct GatedSSMStrategy {
    /// Underlying Gated SSM model
    pub model: GatedSSMBlock,
    /// Threshold for signal generation (distance from 0.5)
    pub threshold: f64,
    /// Feature window buffer
    pub window: Vec<Vec<f64>>,
    /// Maximum window size
    pub window_size: usize,
}

impl GatedSSMStrategy {
    /// Create a new Gated SSM trading strategy.
    ///
    /// # Arguments
    /// * `d_model` - Feature dimension
    /// * `d_state` - SSM state dimension
    /// * `threshold` - Signal threshold (e.g., 0.05)
    /// * `window_size` - Lookback window size
    pub fn new(d_model: usize, d_state: usize, threshold: f64, window_size: usize) -> Self {
        GatedSSMStrategy {
            model: GatedSSMBlock::new(d_model, d_state, 2),
            threshold,
            window: Vec::new(),
            window_size,
        }
    }

    /// Feed a new feature vector and generate a trading signal.
    ///
    /// The model processes the feature vector through the Gated SSM
    /// and uses the first output dimension as a directional score.
    pub fn on_bar(&mut self, features: &[f64]) -> TradingSignal {
        let output = self.model.step(features);

        // Use first output dimension as prediction score
        let score = sigmoid(output[0]);

        let direction = if score > 0.5 + self.threshold {
            SignalDirection::Long
        } else if score < 0.5 - self.threshold {
            SignalDirection::Short
        } else {
            SignalDirection::Neutral
        };

        TradingSignal {
            timestamp: 0,
            direction,
            confidence: (score - 0.5).abs() * 2.0,
        }
    }

    /// Reset the strategy state.
    pub fn reset(&mut self) {
        self.model.reset();
        self.window.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_basic() {
        let mut strategy = GatedSSMStrategy::new(4, 16, 0.05, 60);
        let features = vec![0.01, 0.02, 0.5, -0.01];
        let signal = strategy.on_bar(&features);
        assert!(signal.confidence >= 0.0 && signal.confidence <= 1.0);
    }

    #[test]
    fn test_strategy_sequence() {
        let mut strategy = GatedSSMStrategy::new(4, 16, 0.05, 60);
        let mut signals = Vec::new();
        for i in 0..100 {
            let features = vec![
                (i as f64 * 0.1).sin() * 0.02,
                0.01,
                0.5,
                -0.005,
            ];
            signals.push(strategy.on_bar(&features));
        }
        assert_eq!(signals.len(), 100);
    }
}
