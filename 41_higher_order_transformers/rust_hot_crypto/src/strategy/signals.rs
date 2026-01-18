//! Trading signal generation
//!
//! Converts model predictions into trading signals with risk management

use serde::{Deserialize, Serialize};

use crate::model::{MovementClass, PredictionResult};

/// Trading signal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Signal {
    /// Buy signal
    Buy,
    /// Sell signal
    Sell,
    /// Hold (do nothing)
    Hold,
}

impl Signal {
    /// Get display string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Buy => "BUY",
            Self::Sell => "SELL",
            Self::Hold => "HOLD",
        }
    }

    /// Check if this is an action signal
    pub fn is_action(&self) -> bool {
        !matches!(self, Self::Hold)
    }
}

/// Signal generation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalResult {
    /// Symbol
    pub symbol: String,
    /// Generated signal
    pub signal: Signal,
    /// Prediction confidence
    pub confidence: f64,
    /// Underlying prediction
    pub prediction: MovementClass,
    /// Suggested position size (0.0 to 1.0)
    pub position_size: f64,
    /// Reason for the signal
    pub reason: String,
}

/// Signal generator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalGeneratorConfig {
    /// Minimum confidence to act
    pub min_confidence: f64,
    /// Maximum position size
    pub max_position: f64,
    /// Base position size
    pub base_position: f64,
    /// Scale position by confidence
    pub scale_by_confidence: bool,
}

impl Default for SignalGeneratorConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            max_position: 0.25,
            base_position: 0.1,
            scale_by_confidence: true,
        }
    }
}

/// Trading signal generator
pub struct SignalGenerator {
    /// Configuration
    pub config: SignalGeneratorConfig,
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl SignalGenerator {
    /// Create a new signal generator with default config
    pub fn new() -> Self {
        Self {
            config: SignalGeneratorConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: SignalGeneratorConfig) -> Self {
        Self { config }
    }

    /// Generate a trading signal from a prediction
    pub fn generate(&self, prediction: &PredictionResult) -> SignalResult {
        let (signal, reason) = self.determine_signal(prediction);
        let position_size = self.calculate_position_size(prediction, &signal);

        SignalResult {
            symbol: prediction.symbol.clone(),
            signal,
            confidence: prediction.confidence,
            prediction: prediction.prediction,
            position_size,
            reason,
        }
    }

    /// Determine the signal based on prediction
    fn determine_signal(&self, prediction: &PredictionResult) -> (Signal, String) {
        // Check confidence threshold
        if prediction.confidence < self.config.min_confidence {
            return (
                Signal::Hold,
                format!(
                    "Confidence {:.1}% below threshold {:.1}%",
                    prediction.confidence * 100.0,
                    self.config.min_confidence * 100.0
                ),
            );
        }

        match prediction.prediction {
            MovementClass::Up => (
                Signal::Buy,
                format!(
                    "Bullish prediction with {:.1}% confidence",
                    prediction.confidence * 100.0
                ),
            ),
            MovementClass::Down => (
                Signal::Sell,
                format!(
                    "Bearish prediction with {:.1}% confidence",
                    prediction.confidence * 100.0
                ),
            ),
            MovementClass::Neutral => (
                Signal::Hold,
                "Neutral prediction - no clear direction".to_string(),
            ),
        }
    }

    /// Calculate position size based on confidence
    fn calculate_position_size(&self, prediction: &PredictionResult, signal: &Signal) -> f64 {
        if *signal == Signal::Hold {
            return 0.0;
        }

        let mut size = self.config.base_position;

        if self.config.scale_by_confidence {
            // Scale by excess confidence above threshold
            let excess = prediction.confidence - self.config.min_confidence;
            let scale_factor = 1.0 + (excess / (1.0 - self.config.min_confidence));
            size *= scale_factor;
        }

        // Cap at maximum
        size.min(self.config.max_position)
    }

    /// Generate signals for multiple predictions
    pub fn generate_batch(&self, predictions: &[PredictionResult]) -> Vec<SignalResult> {
        predictions.iter().map(|p| self.generate(p)).collect()
    }

    /// Filter signals to only actionable ones
    pub fn actionable_signals(&self, signals: &[SignalResult]) -> Vec<&SignalResult> {
        signals.iter().filter(|s| s.signal.is_action()).collect()
    }

    /// Get the strongest buy signal
    pub fn strongest_buy<'a>(&self, signals: &'a [SignalResult]) -> Option<&'a SignalResult> {
        signals
            .iter()
            .filter(|s| s.signal == Signal::Buy)
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
    }

    /// Get the strongest sell signal
    pub fn strongest_sell<'a>(&self, signals: &'a [SignalResult]) -> Option<&'a SignalResult> {
        signals
            .iter()
            .filter(|s| s.signal == Signal::Sell)
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
    }
}

/// Risk management rules
pub struct RiskManager {
    /// Maximum portfolio exposure
    pub max_exposure: f64,
    /// Stop loss percentage
    pub stop_loss: f64,
    /// Take profit percentage
    pub take_profit: f64,
    /// Maximum correlation between positions
    pub max_correlation: f64,
}

impl Default for RiskManager {
    fn default() -> Self {
        Self {
            max_exposure: 0.8,
            stop_loss: 0.03,
            take_profit: 0.05,
            max_correlation: 0.7,
        }
    }
}

impl RiskManager {
    /// Check if a new position would exceed risk limits
    pub fn can_open_position(&self, current_exposure: f64, new_position_size: f64) -> bool {
        current_exposure + new_position_size <= self.max_exposure
    }

    /// Calculate adjusted position size
    pub fn adjust_position_size(&self, desired_size: f64, current_exposure: f64) -> f64 {
        let available = (self.max_exposure - current_exposure).max(0.0);
        desired_size.min(available)
    }

    /// Check if stop loss was hit
    pub fn is_stop_loss(&self, entry_price: f64, current_price: f64, is_long: bool) -> bool {
        if is_long {
            (entry_price - current_price) / entry_price >= self.stop_loss
        } else {
            (current_price - entry_price) / entry_price >= self.stop_loss
        }
    }

    /// Check if take profit was hit
    pub fn is_take_profit(&self, entry_price: f64, current_price: f64, is_long: bool) -> bool {
        if is_long {
            (current_price - entry_price) / entry_price >= self.take_profit
        } else {
            (entry_price - current_price) / entry_price >= self.take_profit
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_generation() {
        let generator = SignalGenerator::new();

        let prediction = PredictionResult {
            symbol: "BTCUSDT".to_string(),
            prediction: MovementClass::Up,
            confidence: 0.75,
            probabilities: [0.15, 0.10, 0.75],
        };

        let signal = generator.generate(&prediction);

        assert_eq!(signal.signal, Signal::Buy);
        assert!(signal.position_size > 0.0);
    }

    #[test]
    fn test_low_confidence_hold() {
        let generator = SignalGenerator::new();

        let prediction = PredictionResult {
            symbol: "BTCUSDT".to_string(),
            prediction: MovementClass::Up,
            confidence: 0.45, // Below threshold
            probabilities: [0.25, 0.30, 0.45],
        };

        let signal = generator.generate(&prediction);

        assert_eq!(signal.signal, Signal::Hold);
        assert_eq!(signal.position_size, 0.0);
    }

    #[test]
    fn test_risk_manager_stop_loss() {
        let rm = RiskManager::default();

        // Long position
        assert!(rm.is_stop_loss(100.0, 96.0, true)); // 4% loss
        assert!(!rm.is_stop_loss(100.0, 98.0, true)); // 2% loss

        // Short position
        assert!(rm.is_stop_loss(100.0, 104.0, false)); // 4% loss
        assert!(!rm.is_stop_loss(100.0, 102.0, false)); // 2% loss
    }
}
