//! Signal generation from model predictions
//!
//! This module converts model outputs into trading signals.

use serde::{Deserialize, Serialize};

/// Trading signal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Signal {
    /// Buy signal
    Buy,
    /// Sell signal
    Sell,
    /// Hold/no action
    Hold,
}

impl Signal {
    /// Convert from class prediction
    pub fn from_class(class: i64) -> Self {
        match class {
            0 => Signal::Sell,  // Bearish
            1 => Signal::Hold,  // Neutral
            2 => Signal::Buy,   // Bullish
            _ => Signal::Hold,
        }
    }

    /// Check if signal is actionable
    pub fn is_actionable(&self) -> bool {
        matches!(self, Signal::Buy | Signal::Sell)
    }
}

/// Signal with confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalWithConfidence {
    pub signal: Signal,
    pub confidence: f64,
    pub timestamp: i64,
    pub price: f64,
}

/// Signal generator from model predictions
#[derive(Debug, Clone)]
pub struct SignalGenerator {
    /// Minimum confidence threshold
    min_confidence: f64,
    /// Whether to invert signals (for contrarian strategy)
    invert: bool,
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new(min_confidence: f64) -> Self {
        Self {
            min_confidence,
            invert: false,
        }
    }

    /// Create a contrarian signal generator
    pub fn contrarian(min_confidence: f64) -> Self {
        Self {
            min_confidence,
            invert: true,
        }
    }

    /// Generate signal from prediction
    ///
    /// # Arguments
    /// * `probabilities` - Class probabilities [bearish, neutral, bullish]
    /// * `timestamp` - Current timestamp
    /// * `price` - Current price
    pub fn generate(&self, probabilities: &[f64; 3], timestamp: i64, price: f64) -> SignalWithConfidence {
        let (class, confidence) = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, &c)| (i as i64, c))
            .unwrap_or((1, 0.0));

        let mut signal = if confidence >= self.min_confidence {
            Signal::from_class(class)
        } else {
            Signal::Hold
        };

        // Invert if contrarian
        if self.invert && signal.is_actionable() {
            signal = match signal {
                Signal::Buy => Signal::Sell,
                Signal::Sell => Signal::Buy,
                _ => signal,
            };
        }

        SignalWithConfidence {
            signal,
            confidence,
            timestamp,
            price,
        }
    }

    /// Generate signal from ensemble predictions
    pub fn generate_from_ensemble(
        &self,
        ensemble_probs: &[[f64; 3]],
        timestamp: i64,
        price: f64,
    ) -> SignalWithConfidence {
        // Average probabilities across ensemble
        let mut avg_probs = [0.0; 3];
        for probs in ensemble_probs {
            for (i, &p) in probs.iter().enumerate() {
                avg_probs[i] += p / ensemble_probs.len() as f64;
            }
        }

        // Calculate ensemble uncertainty (std of predictions)
        let mut uncertainty = 0.0;
        for probs in ensemble_probs {
            for (i, &p) in probs.iter().enumerate() {
                uncertainty += (p - avg_probs[i]).powi(2);
            }
        }
        uncertainty = (uncertainty / (ensemble_probs.len() * 3) as f64).sqrt();

        // Reduce confidence based on uncertainty
        let adjusted_probs = avg_probs.map(|p| p * (1.0 - uncertainty));

        self.generate(
            &[adjusted_probs[0], adjusted_probs[1], adjusted_probs[2]],
            timestamp,
            price,
        )
    }
}

/// Complete trading strategy combining signal generation with rules
#[derive(Debug, Clone)]
pub struct TradingStrategy {
    /// Signal generator
    signal_generator: SignalGenerator,
    /// Maximum number of consecutive trades in same direction
    max_consecutive: usize,
    /// Cooldown period after trade (in candles)
    cooldown: usize,
    /// Current consecutive trades counter
    consecutive_count: usize,
    /// Last signal direction
    last_signal: Option<Signal>,
    /// Cooldown counter
    cooldown_counter: usize,
}

impl TradingStrategy {
    /// Create a new trading strategy
    pub fn new(min_confidence: f64, risk_per_trade: f64) -> Self {
        Self {
            signal_generator: SignalGenerator::new(min_confidence),
            max_consecutive: 3,
            cooldown: 2,
            consecutive_count: 0,
            last_signal: None,
            cooldown_counter: 0,
        }
    }

    /// Process prediction and get trading action
    pub fn process(
        &mut self,
        probabilities: &[f64; 3],
        timestamp: i64,
        price: f64,
    ) -> SignalWithConfidence {
        // Check cooldown
        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            return SignalWithConfidence {
                signal: Signal::Hold,
                confidence: 0.0,
                timestamp,
                price,
            };
        }

        let signal = self.signal_generator.generate(probabilities, timestamp, price);

        // Check consecutive trades limit
        if signal.signal.is_actionable() {
            if self.last_signal == Some(signal.signal) {
                self.consecutive_count += 1;
                if self.consecutive_count >= self.max_consecutive {
                    return SignalWithConfidence {
                        signal: Signal::Hold,
                        confidence: signal.confidence,
                        timestamp,
                        price,
                    };
                }
            } else {
                self.consecutive_count = 1;
            }

            self.last_signal = Some(signal.signal);
            self.cooldown_counter = self.cooldown;
        }

        signal
    }

    /// Reset strategy state
    pub fn reset(&mut self) {
        self.consecutive_count = 0;
        self.last_signal = None;
        self.cooldown_counter = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_from_class() {
        assert_eq!(Signal::from_class(0), Signal::Sell);
        assert_eq!(Signal::from_class(1), Signal::Hold);
        assert_eq!(Signal::from_class(2), Signal::Buy);
    }

    #[test]
    fn test_signal_generator() {
        let generator = SignalGenerator::new(0.6);

        // High confidence bullish
        let signal = generator.generate(&[0.1, 0.1, 0.8], 1000, 50000.0);
        assert_eq!(signal.signal, Signal::Buy);
        assert!(signal.confidence >= 0.6);

        // Low confidence
        let signal = generator.generate(&[0.4, 0.3, 0.3], 1000, 50000.0);
        assert_eq!(signal.signal, Signal::Hold);
    }

    #[test]
    fn test_contrarian() {
        let generator = SignalGenerator::contrarian(0.6);

        let signal = generator.generate(&[0.1, 0.1, 0.8], 1000, 50000.0);
        assert_eq!(signal.signal, Signal::Sell); // Inverted from Buy
    }
}
