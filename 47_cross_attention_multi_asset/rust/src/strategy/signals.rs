//! Signal generation from model predictions
//!
//! Converts model outputs to trading signals.

use super::Action;
use serde::{Deserialize, Serialize};

/// Trading signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Asset symbol
    pub symbol: String,
    /// Recommended action
    pub action: Action,
    /// Signal strength [0, 1]
    pub strength: f64,
    /// Target portfolio weight
    pub target_weight: f64,
    /// Confidence score
    pub confidence: f64,
}

impl Signal {
    /// Create a new signal
    pub fn new(
        symbol: &str,
        action: Action,
        strength: f64,
        target_weight: f64,
        confidence: f64,
    ) -> Self {
        Self {
            symbol: symbol.to_string(),
            action,
            strength: strength.clamp(0.0, 1.0),
            target_weight: target_weight.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

/// Signal generator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalConfig {
    /// Minimum confidence to generate signal
    pub min_confidence: f64,
    /// Threshold for buy signal
    pub buy_threshold: f64,
    /// Threshold for sell signal
    pub sell_threshold: f64,
    /// Maximum position size per asset
    pub max_position: f64,
    /// Allow short positions
    pub allow_short: bool,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            buy_threshold: 0.0,
            sell_threshold: 0.0,
            max_position: 0.5,
            allow_short: false,
        }
    }
}

/// Signal generator
pub struct SignalGenerator {
    config: SignalConfig,
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new(config: SignalConfig) -> Self {
        Self { config }
    }

    /// Generate signals from model predictions
    ///
    /// # Arguments
    /// * `predictions` - Model predictions (portfolio weights or returns)
    /// * `symbols` - Asset symbols
    /// * `confidence` - Optional confidence scores per asset
    ///
    /// # Returns
    /// Vector of trading signals
    pub fn generate(
        &self,
        predictions: &[f64],
        symbols: &[String],
        confidence: Option<&[f64]>,
    ) -> Vec<Signal> {
        assert_eq!(
            predictions.len(),
            symbols.len(),
            "Predictions and symbols must have same length"
        );

        let default_confidence = vec![1.0; predictions.len()];
        let conf = confidence.unwrap_or(&default_confidence);

        predictions
            .iter()
            .zip(symbols.iter())
            .zip(conf.iter())
            .map(|((&pred, symbol), &c)| {
                let (action, strength) = self.determine_action(pred);
                let target_weight = self.calculate_target_weight(pred, c);

                Signal::new(symbol, action, strength, target_weight, c)
            })
            .filter(|s| s.confidence >= self.config.min_confidence)
            .collect()
    }

    /// Generate signals from portfolio weights
    ///
    /// # Arguments
    /// * `weights` - Portfolio weights (should sum to 1)
    /// * `symbols` - Asset symbols
    ///
    /// # Returns
    /// Vector of trading signals
    pub fn generate_from_weights(
        &self,
        weights: &[f64],
        symbols: &[String],
    ) -> Vec<Signal> {
        assert_eq!(
            weights.len(),
            symbols.len(),
            "Weights and symbols must have same length"
        );

        // Normalize weights if they don't sum to 1
        let total: f64 = weights.iter().sum();
        let normalized: Vec<f64> = if (total - 1.0).abs() > 1e-6 && total > 0.0 {
            weights.iter().map(|w| w / total).collect()
        } else {
            weights.to_vec()
        };

        // Find equal weight baseline
        let equal_weight = 1.0 / weights.len() as f64;

        normalized
            .iter()
            .zip(symbols.iter())
            .map(|(&weight, symbol)| {
                // Action based on deviation from equal weight
                let deviation = weight - equal_weight;
                let (action, strength) = if deviation > self.config.buy_threshold {
                    (Action::Buy, deviation.abs().min(1.0))
                } else if deviation < -self.config.sell_threshold {
                    if self.config.allow_short {
                        (Action::Sell, deviation.abs().min(1.0))
                    } else {
                        (Action::Hold, 0.0)
                    }
                } else {
                    (Action::Hold, 0.0)
                };

                // Target weight capped by max position
                let target = weight.min(self.config.max_position);

                Signal::new(symbol, action, strength, target, 1.0)
            })
            .collect()
    }

    /// Determine action from prediction
    fn determine_action(&self, prediction: f64) -> (Action, f64) {
        if prediction > self.config.buy_threshold {
            (Action::Buy, prediction.min(1.0))
        } else if prediction < -self.config.sell_threshold {
            if self.config.allow_short {
                (Action::Sell, prediction.abs().min(1.0))
            } else {
                (Action::Hold, 0.0)
            }
        } else {
            (Action::Hold, 0.0)
        }
    }

    /// Calculate target weight
    fn calculate_target_weight(&self, prediction: f64, confidence: f64) -> f64 {
        let base_weight = prediction.max(0.0);
        let adjusted = base_weight * confidence;
        adjusted.min(self.config.max_position)
    }
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new(SignalConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_creation() {
        let signal = Signal::new("BTCUSDT", Action::Buy, 0.8, 0.3, 0.9);

        assert_eq!(signal.symbol, "BTCUSDT");
        assert_eq!(signal.action, Action::Buy);
        assert_eq!(signal.strength, 0.8);
        assert_eq!(signal.target_weight, 0.3);
        assert_eq!(signal.confidence, 0.9);
    }

    #[test]
    fn test_signal_clamping() {
        let signal = Signal::new("BTCUSDT", Action::Buy, 1.5, -0.1, 2.0);

        assert_eq!(signal.strength, 1.0);
        assert_eq!(signal.target_weight, 0.0);
        assert_eq!(signal.confidence, 1.0);
    }

    #[test]
    fn test_signal_generator() {
        let config = SignalConfig {
            min_confidence: 0.5,
            buy_threshold: 0.01,
            sell_threshold: 0.01,
            max_position: 0.5,
            allow_short: false,
        };

        let generator = SignalGenerator::new(config);
        let predictions = vec![0.05, -0.02, 0.0];
        let symbols = vec!["BTC".to_string(), "ETH".to_string(), "SOL".to_string()];

        let signals = generator.generate(&predictions, &symbols, None);

        assert_eq!(signals.len(), 3);
        assert_eq!(signals[0].action, Action::Buy);
        assert_eq!(signals[1].action, Action::Hold); // No short allowed
        assert_eq!(signals[2].action, Action::Hold);
    }

    #[test]
    fn test_generate_from_weights() {
        let generator = SignalGenerator::default();
        let weights = vec![0.5, 0.3, 0.2];
        let symbols = vec!["BTC".to_string(), "ETH".to_string(), "SOL".to_string()];

        let signals = generator.generate_from_weights(&weights, &symbols);

        assert_eq!(signals.len(), 3);

        // BTC has highest weight (above equal ~0.33)
        assert_eq!(signals[0].action, Action::Buy);

        // Target weights should be capped at max_position
        for signal in &signals {
            assert!(signal.target_weight <= 0.5);
        }
    }

    #[test]
    fn test_weight_normalization() {
        let generator = SignalGenerator::default();
        let weights = vec![1.0, 0.5, 0.5]; // Sum = 2.0
        let symbols = vec!["BTC".to_string(), "ETH".to_string(), "SOL".to_string()];

        let signals = generator.generate_from_weights(&weights, &symbols);

        // After normalization: 0.5, 0.25, 0.25
        assert!(signals[0].target_weight > signals[1].target_weight);
    }
}
