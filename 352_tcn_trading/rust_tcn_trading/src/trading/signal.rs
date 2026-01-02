//! Trading Signal Generation

use chrono::{DateTime, Utc};
use ndarray::Array1;

use crate::features::FeatureMatrix;
use crate::tcn::TCN;

/// Signal type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    /// Long/Buy signal
    Long,
    /// Short/Sell signal
    Short,
    /// No signal / Hold
    Neutral,
}

impl SignalType {
    /// Convert to position multiplier
    pub fn to_position(&self) -> f64 {
        match self {
            SignalType::Long => 1.0,
            SignalType::Short => -1.0,
            SignalType::Neutral => 0.0,
        }
    }

    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            SignalType::Long => "LONG",
            SignalType::Short => "SHORT",
            SignalType::Neutral => "NEUTRAL",
        }
    }
}

/// Trading signal with metadata
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Signal type
    pub signal_type: SignalType,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Suggested position size (0.0 to 1.0)
    pub position_size: f64,
    /// Predicted return
    pub predicted_return: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Model prediction probabilities
    pub probabilities: Option<Array1<f64>>,
}

impl TradingSignal {
    /// Create a new trading signal
    pub fn new(
        signal_type: SignalType,
        confidence: f64,
        position_size: f64,
        predicted_return: f64,
    ) -> Self {
        Self {
            signal_type,
            confidence,
            position_size,
            predicted_return,
            timestamp: Utc::now(),
            probabilities: None,
        }
    }

    /// Create a neutral signal
    pub fn neutral() -> Self {
        Self::new(SignalType::Neutral, 0.0, 0.0, 0.0)
    }

    /// Check if signal is actionable
    pub fn is_actionable(&self) -> bool {
        self.signal_type != SignalType::Neutral && self.confidence > 0.5
    }
}

/// Signal generator using TCN model
#[derive(Debug)]
pub struct SignalGenerator {
    /// TCN model
    pub model: TCN,
    /// Threshold for long signals
    pub threshold_long: f64,
    /// Threshold for short signals
    pub threshold_short: f64,
    /// Minimum confidence required
    pub min_confidence: f64,
    /// Position sizing method
    pub position_sizing: PositionSizing,
}

/// Position sizing method
#[derive(Debug, Clone, Copy)]
pub enum PositionSizing {
    /// Fixed position size
    Fixed(f64),
    /// Scale by confidence
    ConfidenceScaled,
    /// Kelly criterion based
    Kelly { max_fraction: f64 },
}

impl Default for PositionSizing {
    fn default() -> Self {
        PositionSizing::ConfidenceScaled
    }
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new(model: TCN, threshold_long: f64, threshold_short: f64) -> Self {
        Self {
            model,
            threshold_long,
            threshold_short,
            min_confidence: 0.55,
            position_sizing: PositionSizing::default(),
        }
    }

    /// Set minimum confidence threshold
    pub fn with_min_confidence(mut self, min_confidence: f64) -> Self {
        self.min_confidence = min_confidence;
        self
    }

    /// Set position sizing method
    pub fn with_position_sizing(mut self, method: PositionSizing) -> Self {
        self.position_sizing = method;
        self
    }

    /// Generate trading signal from features
    pub fn generate_signal(&self, features: &FeatureMatrix) -> TradingSignal {
        let input = &features.data;

        // Get model prediction
        let proba = self.model.predict_proba(input);

        // Determine signal based on probabilities
        // Assuming: index 0 = Down, index 1 = Neutral, index 2 = Up
        let (signal_type, confidence, predicted_return) = if proba.len() >= 3 {
            let prob_up = proba[2];
            let prob_down = proba[0];
            let prob_neutral = proba[1];

            if prob_up > self.threshold_long && prob_up > prob_down {
                (SignalType::Long, prob_up, prob_up - prob_down)
            } else if prob_down > self.threshold_short && prob_down > prob_up {
                (SignalType::Short, prob_down, prob_down - prob_up)
            } else {
                (SignalType::Neutral, prob_neutral, 0.0)
            }
        } else if proba.len() == 2 {
            // Binary classification
            let prob_up = proba[1];
            let prob_down = proba[0];

            if prob_up > self.threshold_long {
                (SignalType::Long, prob_up, prob_up - 0.5)
            } else if prob_down > self.threshold_short {
                (SignalType::Short, prob_down, prob_down - 0.5)
            } else {
                (SignalType::Neutral, 0.5, 0.0)
            }
        } else {
            // Regression output
            let pred = proba[0];
            if pred > 0.01 {
                (SignalType::Long, pred.abs().min(1.0), pred)
            } else if pred < -0.01 {
                (SignalType::Short, pred.abs().min(1.0), pred)
            } else {
                (SignalType::Neutral, 0.0, pred)
            }
        };

        // Calculate position size
        let position_size = if confidence >= self.min_confidence {
            self.calculate_position_size(confidence, predicted_return)
        } else {
            0.0
        };

        TradingSignal {
            signal_type: if position_size > 0.0 { signal_type } else { SignalType::Neutral },
            confidence,
            position_size,
            predicted_return,
            timestamp: Utc::now(),
            probabilities: Some(proba),
        }
    }

    /// Generate signals for a batch of feature windows
    pub fn generate_signals(&self, features: &[FeatureMatrix]) -> Vec<TradingSignal> {
        features.iter().map(|f| self.generate_signal(f)).collect()
    }

    /// Calculate position size based on sizing method
    fn calculate_position_size(&self, confidence: f64, predicted_return: f64) -> f64 {
        match self.position_sizing {
            PositionSizing::Fixed(size) => size,
            PositionSizing::ConfidenceScaled => {
                // Scale position by confidence
                let base_size = 0.1;
                let confidence_factor = (confidence - 0.5) * 2.0; // 0 to 1
                base_size * (1.0 + confidence_factor).min(2.0)
            }
            PositionSizing::Kelly { max_fraction } => {
                // Simplified Kelly criterion
                // f* = (p * b - q) / b
                // where p = probability of win, q = 1-p, b = win/loss ratio
                let win_prob = confidence;
                let lose_prob = 1.0 - confidence;
                let win_loss_ratio = predicted_return.abs().max(0.01) / 0.01; // Assume 1% stop loss

                let kelly = (win_prob * win_loss_ratio - lose_prob) / win_loss_ratio;
                kelly.max(0.0).min(max_fraction)
            }
        }
    }
}

/// Simple rules-based signal generator
#[derive(Debug)]
pub struct RulesBasedGenerator {
    /// RSI overbought level
    pub rsi_overbought: f64,
    /// RSI oversold level
    pub rsi_oversold: f64,
    /// MACD confirmation required
    pub require_macd_confirm: bool,
}

impl Default for RulesBasedGenerator {
    fn default() -> Self {
        Self {
            rsi_overbought: 70.0,
            rsi_oversold: 30.0,
            require_macd_confirm: true,
        }
    }
}

impl RulesBasedGenerator {
    /// Generate signal from technical indicators
    pub fn generate_signal(&self, features: &FeatureMatrix) -> TradingSignal {
        let rsi = features.get_feature("rsi_14");
        let macd = features.get_feature("macd");
        let macd_signal = features.get_feature("macd_signal");

        let (rsi_val, macd_val, signal_val) = match (rsi, macd, macd_signal) {
            (Some(r), Some(m), Some(s)) => {
                let last = r.len() - 1;
                (r[last], m[last], s[last])
            }
            _ => return TradingSignal::neutral(),
        };

        // Check for NaN
        if rsi_val.is_nan() || macd_val.is_nan() || signal_val.is_nan() {
            return TradingSignal::neutral();
        }

        let macd_bullish = macd_val > signal_val;
        let macd_bearish = macd_val < signal_val;

        let signal = if rsi_val < self.rsi_oversold {
            if !self.require_macd_confirm || macd_bullish {
                SignalType::Long
            } else {
                SignalType::Neutral
            }
        } else if rsi_val > self.rsi_overbought {
            if !self.require_macd_confirm || macd_bearish {
                SignalType::Short
            } else {
                SignalType::Neutral
            }
        } else {
            SignalType::Neutral
        };

        let confidence = match signal {
            SignalType::Long => (self.rsi_oversold - rsi_val).abs() / self.rsi_oversold,
            SignalType::Short => (rsi_val - self.rsi_overbought) / (100.0 - self.rsi_overbought),
            SignalType::Neutral => 0.0,
        };

        TradingSignal::new(signal, confidence.min(1.0), confidence * 0.1, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tcn::TCNConfig;

    #[test]
    fn test_signal_type() {
        assert_eq!(SignalType::Long.to_position(), 1.0);
        assert_eq!(SignalType::Short.to_position(), -1.0);
        assert_eq!(SignalType::Neutral.to_position(), 0.0);
    }

    #[test]
    fn test_trading_signal_creation() {
        let signal = TradingSignal::new(SignalType::Long, 0.75, 0.1, 0.02);
        assert!(signal.is_actionable());

        let neutral = TradingSignal::neutral();
        assert!(!neutral.is_actionable());
    }

    #[test]
    fn test_signal_generator_creation() {
        let config = TCNConfig::default();
        let model = TCN::new(config);
        let generator = SignalGenerator::new(model, 0.6, 0.6);

        assert_eq!(generator.threshold_long, 0.6);
        assert_eq!(generator.threshold_short, 0.6);
    }

    #[test]
    fn test_position_sizing() {
        let config = TCNConfig::default();
        let model = TCN::new(config);
        let generator = SignalGenerator::new(model, 0.6, 0.6)
            .with_position_sizing(PositionSizing::Fixed(0.05));

        let size = generator.calculate_position_size(0.8, 0.02);
        assert_eq!(size, 0.05);
    }
}
