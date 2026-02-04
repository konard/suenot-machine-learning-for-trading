//! Trading signal generation

use chrono::{DateTime, Utc};

/// Signal type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    Buy,
    Sell,
    Hold,
}

impl std::fmt::Display for SignalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalType::Buy => write!(f, "BUY"),
            SignalType::Sell => write!(f, "SELL"),
            SignalType::Hold => write!(f, "HOLD"),
        }
    }
}

/// Trading signal
#[derive(Debug, Clone)]
pub struct Signal {
    pub signal_type: SignalType,
    pub confidence: f64,
    pub expected_return: Option<f64>,
    pub timestamp: Option<DateTime<Utc>>,
    pub symbol: Option<String>,
}

impl Signal {
    /// Create a new signal
    pub fn new(signal_type: SignalType, confidence: f64) -> Self {
        Self {
            signal_type,
            confidence,
            expected_return: None,
            timestamp: None,
            symbol: None,
        }
    }

    /// Create a signal with expected return
    pub fn with_return(mut self, expected_return: f64) -> Self {
        self.expected_return = Some(expected_return);
        self
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Set symbol
    pub fn with_symbol(mut self, symbol: &str) -> Self {
        self.symbol = Some(symbol.to_string());
        self
    }

    /// Check if this is an actionable signal (not HOLD)
    pub fn is_actionable(&self) -> bool {
        self.signal_type != SignalType::Hold
    }
}

/// Signal generator configuration
#[derive(Debug, Clone)]
pub struct SignalGeneratorConfig {
    /// Threshold for buy signals
    pub buy_threshold: f64,
    /// Threshold for sell signals
    pub sell_threshold: f64,
    /// Minimum expected return for actionable signals
    pub min_expected_return: f64,
    /// Whether to use expected return filtering
    pub use_return_filter: bool,
}

impl Default for SignalGeneratorConfig {
    fn default() -> Self {
        Self {
            buy_threshold: 0.6,
            sell_threshold: 0.6,
            min_expected_return: 0.005,
            use_return_filter: true,
        }
    }
}

/// Signal generator
#[derive(Debug, Clone)]
pub struct SignalGenerator {
    pub config: SignalGeneratorConfig,
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new(config: SignalGeneratorConfig) -> Self {
        Self { config }
    }

    /// Create with simple thresholds
    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            config: SignalGeneratorConfig {
                buy_threshold: threshold,
                sell_threshold: threshold,
                ..Default::default()
            },
        }
    }

    /// Generate signal from probabilities
    pub fn generate(
        &self,
        prob_down: f64,
        prob_neutral: f64,
        prob_up: f64,
        magnitude: Option<f64>,
    ) -> Signal {
        let signal = if prob_up > self.config.buy_threshold {
            if self.config.use_return_filter {
                if let Some(mag) = magnitude {
                    if mag > self.config.min_expected_return {
                        Signal::new(SignalType::Buy, prob_up).with_return(mag)
                    } else {
                        Signal::new(SignalType::Hold, 0.0)
                    }
                } else {
                    Signal::new(SignalType::Buy, prob_up)
                }
            } else {
                Signal::new(SignalType::Buy, prob_up)
            }
        } else if prob_down > self.config.sell_threshold {
            if self.config.use_return_filter {
                if let Some(mag) = magnitude {
                    if mag < -self.config.min_expected_return {
                        Signal::new(SignalType::Sell, prob_down).with_return(mag)
                    } else {
                        Signal::new(SignalType::Hold, 0.0)
                    }
                } else {
                    Signal::new(SignalType::Sell, prob_down)
                }
            } else {
                Signal::new(SignalType::Sell, prob_down)
            }
        } else {
            Signal::new(SignalType::Hold, prob_neutral)
        };

        signal
    }

    /// Generate signals from a batch of predictions
    pub fn generate_batch(
        &self,
        predictions: &[(f64, f64, f64, Option<f64>)],
    ) -> Vec<Signal> {
        predictions
            .iter()
            .map(|(down, neutral, up, mag)| self.generate(*down, *neutral, *up, *mag))
            .collect()
    }
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new(SignalGeneratorConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_creation() {
        let signal = Signal::new(SignalType::Buy, 0.7);
        assert_eq!(signal.signal_type, SignalType::Buy);
        assert_eq!(signal.confidence, 0.7);
        assert!(signal.is_actionable());
    }

    #[test]
    fn test_hold_not_actionable() {
        let signal = Signal::new(SignalType::Hold, 0.5);
        assert!(!signal.is_actionable());
    }

    #[test]
    fn test_signal_generator() {
        let generator = SignalGenerator::with_threshold(0.6);

        // High probability up
        let signal = generator.generate(0.1, 0.2, 0.7, None);
        assert_eq!(signal.signal_type, SignalType::Buy);

        // High probability down
        let signal = generator.generate(0.7, 0.2, 0.1, None);
        assert_eq!(signal.signal_type, SignalType::Sell);

        // Neutral
        let signal = generator.generate(0.3, 0.4, 0.3, None);
        assert_eq!(signal.signal_type, SignalType::Hold);
    }

    #[test]
    fn test_return_filter() {
        let config = SignalGeneratorConfig {
            buy_threshold: 0.6,
            min_expected_return: 0.01,
            use_return_filter: true,
            ..Default::default()
        };
        let generator = SignalGenerator::new(config);

        // High prob but low expected return
        let signal = generator.generate(0.1, 0.2, 0.7, Some(0.005));
        assert_eq!(signal.signal_type, SignalType::Hold);

        // High prob and high expected return
        let signal = generator.generate(0.1, 0.2, 0.7, Some(0.02));
        assert_eq!(signal.signal_type, SignalType::Buy);
    }
}
