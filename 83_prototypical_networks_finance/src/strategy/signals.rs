//! Trading signal generation based on regime classification
//!
//! Converts market regime classifications into actionable trading signals.

use crate::data::{MarketRegime, TradingBias};
use crate::strategy::ClassificationResult;
use serde::{Deserialize, Serialize};

/// Trading signal type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    /// Strong buy signal
    StrongBuy,
    /// Moderate buy signal
    Buy,
    /// Hold/neutral signal
    Hold,
    /// Moderate sell signal
    Sell,
    /// Strong sell signal
    StrongSell,
}

impl SignalType {
    /// Get the direction multiplier (-1.0 to 1.0)
    pub fn direction(&self) -> f64 {
        match self {
            SignalType::StrongBuy => 1.0,
            SignalType::Buy => 0.5,
            SignalType::Hold => 0.0,
            SignalType::Sell => -0.5,
            SignalType::StrongSell => -1.0,
        }
    }

    /// Check if this is a buy signal
    pub fn is_buy(&self) -> bool {
        matches!(self, SignalType::StrongBuy | SignalType::Buy)
    }

    /// Check if this is a sell signal
    pub fn is_sell(&self) -> bool {
        matches!(self, SignalType::StrongSell | SignalType::Sell)
    }
}

/// A trading signal with associated metadata
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Signal type
    pub signal_type: SignalType,
    /// Detected market regime
    pub regime: MarketRegime,
    /// Confidence in the classification
    pub confidence: f64,
    /// Suggested position size (0.0 to 1.0)
    pub position_size: f64,
    /// Whether the current market state is unusual
    pub is_unusual: bool,
    /// Reason for the signal
    pub reason: String,
}

impl TradingSignal {
    /// Get the risk-adjusted position size
    pub fn risk_adjusted_size(&self, risk_factor: f64) -> f64 {
        self.position_size * self.confidence * risk_factor
    }
}

/// Configuration for signal generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalConfig {
    /// Minimum confidence to generate a signal (0.0 to 1.0)
    pub min_confidence: f64,
    /// Confidence threshold for strong signals
    pub strong_signal_threshold: f64,
    /// Whether to generate signals during uncertain conditions
    pub allow_uncertain: bool,
    /// Maximum position size (0.0 to 1.0)
    pub max_position_size: f64,
    /// Base position size for weak signals
    pub base_position_size: f64,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.4,
            strong_signal_threshold: 0.7,
            allow_uncertain: false,
            max_position_size: 1.0,
            base_position_size: 0.5,
        }
    }
}

/// Generator for trading signals from regime classifications
pub struct SignalGenerator {
    config: SignalConfig,
    /// Previous signal for trend detection
    previous_signal: Option<TradingSignal>,
}

impl SignalGenerator {
    /// Create a new signal generator with default config
    pub fn new() -> Self {
        Self::with_config(SignalConfig::default())
    }

    /// Create a new signal generator with custom config
    pub fn with_config(config: SignalConfig) -> Self {
        Self {
            config,
            previous_signal: None,
        }
    }

    /// Generate a trading signal from a classification result
    pub fn generate(&mut self, classification: &ClassificationResult) -> TradingSignal {
        let regime = classification.regime;
        let confidence = classification.confidence;
        let is_uncertain = classification.is_uncertain(0.15);
        let is_outlier = classification.is_outlier;

        // Check if we should generate a signal at all
        if confidence < self.config.min_confidence && !self.config.allow_uncertain {
            return self.create_hold_signal(
                regime,
                confidence,
                "Confidence below threshold".to_string(),
            );
        }

        if is_outlier && !self.config.allow_uncertain {
            return self.create_hold_signal(
                regime,
                confidence,
                "Unusual market conditions detected".to_string(),
            );
        }

        // Determine signal type based on regime and confidence
        let (signal_type, reason) = self.determine_signal_type(regime, confidence, is_uncertain);

        // Calculate position size
        let position_size = self.calculate_position_size(signal_type, confidence, is_uncertain);

        let signal = TradingSignal {
            signal_type,
            regime,
            confidence,
            position_size,
            is_unusual: is_outlier || is_uncertain,
            reason,
        };

        self.previous_signal = Some(signal.clone());
        signal
    }

    /// Determine the signal type based on regime
    fn determine_signal_type(
        &self,
        regime: MarketRegime,
        confidence: f64,
        is_uncertain: bool,
    ) -> (SignalType, String) {
        let bias = regime.trading_bias();
        let is_strong = confidence >= self.config.strong_signal_threshold && !is_uncertain;

        match bias {
            TradingBias::StrongLong => {
                if is_strong {
                    (
                        SignalType::StrongBuy,
                        format!("Strong uptrend detected with {:.0}% confidence", confidence * 100.0),
                    )
                } else {
                    (
                        SignalType::Buy,
                        format!("Uptrend detected with {:.0}% confidence", confidence * 100.0),
                    )
                }
            }
            TradingBias::WeakLong => (
                SignalType::Buy,
                format!("Weak uptrend detected with {:.0}% confidence", confidence * 100.0),
            ),
            TradingBias::Neutral => (
                SignalType::Hold,
                format!("Sideways market detected with {:.0}% confidence", confidence * 100.0),
            ),
            TradingBias::WeakShort => (
                SignalType::Sell,
                format!("Weak downtrend detected with {:.0}% confidence", confidence * 100.0),
            ),
            TradingBias::StrongShort => {
                if is_strong {
                    (
                        SignalType::StrongSell,
                        format!("Strong downtrend/crash detected with {:.0}% confidence", confidence * 100.0),
                    )
                } else {
                    (
                        SignalType::Sell,
                        format!("Downtrend detected with {:.0}% confidence", confidence * 100.0),
                    )
                }
            }
        }
    }

    /// Calculate position size based on signal and confidence
    fn calculate_position_size(
        &self,
        signal_type: SignalType,
        confidence: f64,
        is_uncertain: bool,
    ) -> f64 {
        if matches!(signal_type, SignalType::Hold) {
            return 0.0;
        }

        let base_size = match signal_type {
            SignalType::StrongBuy | SignalType::StrongSell => self.config.max_position_size,
            SignalType::Buy | SignalType::Sell => self.config.base_position_size,
            SignalType::Hold => 0.0,
        };

        // Adjust by confidence
        let confidence_adjusted = base_size * confidence;

        // Reduce size if uncertain
        let size = if is_uncertain {
            confidence_adjusted * 0.5
        } else {
            confidence_adjusted
        };

        size.min(self.config.max_position_size)
    }

    /// Create a hold signal
    fn create_hold_signal(&self, regime: MarketRegime, confidence: f64, reason: String) -> TradingSignal {
        TradingSignal {
            signal_type: SignalType::Hold,
            regime,
            confidence,
            position_size: 0.0,
            is_unusual: true,
            reason,
        }
    }

    /// Get the previous signal
    pub fn previous_signal(&self) -> Option<&TradingSignal> {
        self.previous_signal.as_ref()
    }

    /// Check if signal changed from previous
    pub fn signal_changed(&self, current: &TradingSignal) -> bool {
        match &self.previous_signal {
            Some(prev) => prev.signal_type != current.signal_type,
            None => true,
        }
    }

    /// Reset the signal generator state
    pub fn reset(&mut self) {
        self.previous_signal = None;
    }
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_classification(regime: MarketRegime, confidence: f64) -> ClassificationResult {
        let mut probs: Vec<(MarketRegime, f64)> = MarketRegime::all()
            .into_iter()
            .map(|r| {
                if r == regime {
                    (r, confidence)
                } else {
                    (r, (1.0 - confidence) / 4.0)
                }
            })
            .collect();
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        ClassificationResult {
            regime,
            confidence,
            probabilities: probs,
            min_distance: 1.0 - confidence,
            is_outlier: false,
        }
    }

    #[test]
    fn test_signal_generation_strong_uptrend() {
        let mut generator = SignalGenerator::new();
        let classification = create_test_classification(MarketRegime::StrongUptrend, 0.85);

        let signal = generator.generate(&classification);

        assert_eq!(signal.signal_type, SignalType::StrongBuy);
        assert!(signal.position_size > 0.5);
        assert!(!signal.is_unusual);
    }

    #[test]
    fn test_signal_generation_sideways() {
        let mut generator = SignalGenerator::new();
        let classification = create_test_classification(MarketRegime::Sideways, 0.75);

        let signal = generator.generate(&classification);

        assert_eq!(signal.signal_type, SignalType::Hold);
        assert_eq!(signal.position_size, 0.0);
    }

    #[test]
    fn test_signal_generation_low_confidence() {
        let mut generator = SignalGenerator::new();
        let classification = create_test_classification(MarketRegime::StrongUptrend, 0.3);

        let signal = generator.generate(&classification);

        // Low confidence should result in hold
        assert_eq!(signal.signal_type, SignalType::Hold);
        assert!(signal.is_unusual);
    }

    #[test]
    fn test_signal_generation_crash() {
        let mut generator = SignalGenerator::new();
        let classification = create_test_classification(MarketRegime::StrongDowntrend, 0.9);

        let signal = generator.generate(&classification);

        assert_eq!(signal.signal_type, SignalType::StrongSell);
        assert!(signal.position_size > 0.0);
    }

    #[test]
    fn test_signal_type_direction() {
        assert_eq!(SignalType::StrongBuy.direction(), 1.0);
        assert_eq!(SignalType::Buy.direction(), 0.5);
        assert_eq!(SignalType::Hold.direction(), 0.0);
        assert_eq!(SignalType::Sell.direction(), -0.5);
        assert_eq!(SignalType::StrongSell.direction(), -1.0);
    }
}
