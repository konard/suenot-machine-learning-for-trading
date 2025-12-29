//! Trading Signals
//!
//! Signal generation from model predictions

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Trading signal
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Signal {
    /// Strong buy signal
    StrongBuy,
    /// Buy signal
    Buy,
    /// Hold / no action
    Hold,
    /// Sell signal
    Sell,
    /// Strong sell signal
    StrongSell,
}

impl Signal {
    /// Convert to position direction multiplier
    pub fn to_direction(&self) -> f64 {
        match self {
            Signal::StrongBuy => 1.0,
            Signal::Buy => 1.0,
            Signal::Hold => 0.0,
            Signal::Sell => -1.0,
            Signal::StrongSell => -1.0,
        }
    }

    /// Convert to position size multiplier (0.0 to 1.0)
    pub fn to_size(&self) -> f64 {
        match self {
            Signal::StrongBuy => 1.0,
            Signal::Buy => 0.5,
            Signal::Hold => 0.0,
            Signal::Sell => 0.5,
            Signal::StrongSell => 1.0,
        }
    }

    /// Check if this is a long signal
    pub fn is_long(&self) -> bool {
        matches!(self, Signal::StrongBuy | Signal::Buy)
    }

    /// Check if this is a short signal
    pub fn is_short(&self) -> bool {
        matches!(self, Signal::Sell | Signal::StrongSell)
    }

    /// Check if this is a neutral signal
    pub fn is_neutral(&self) -> bool {
        matches!(self, Signal::Hold)
    }
}

/// Signal with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalEvent {
    pub timestamp: DateTime<Utc>,
    pub signal: Signal,
    pub confidence: f64,
    pub prediction: f64,
    pub price: f64,
}

/// Signal generator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalConfig {
    /// Threshold for buy signal (prediction > threshold)
    pub buy_threshold: f64,
    /// Threshold for strong buy signal
    pub strong_buy_threshold: f64,
    /// Threshold for sell signal (prediction < -threshold)
    pub sell_threshold: f64,
    /// Threshold for strong sell signal
    pub strong_sell_threshold: f64,
    /// Minimum confidence to generate signal
    pub min_confidence: f64,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            buy_threshold: 0.001,          // 0.1% expected return
            strong_buy_threshold: 0.005,   // 0.5% expected return
            sell_threshold: -0.001,
            strong_sell_threshold: -0.005,
            min_confidence: 0.5,
        }
    }
}

/// Signal generator from model predictions
pub struct SignalGenerator {
    pub config: SignalConfig,
}

impl SignalGenerator {
    pub fn new(config: SignalConfig) -> Self {
        Self { config }
    }

    /// Generate signal from prediction
    ///
    /// # Arguments
    /// * `prediction` - Model prediction (expected return or probability)
    /// * `confidence` - Model confidence (0.0 to 1.0)
    pub fn generate(&self, prediction: f64, confidence: f64) -> Signal {
        // Check minimum confidence
        if confidence < self.config.min_confidence {
            return Signal::Hold;
        }

        if prediction >= self.config.strong_buy_threshold {
            Signal::StrongBuy
        } else if prediction >= self.config.buy_threshold {
            Signal::Buy
        } else if prediction <= self.config.strong_sell_threshold {
            Signal::StrongSell
        } else if prediction <= self.config.sell_threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    /// Generate signal from binary classification (probability of up move)
    pub fn generate_from_probability(&self, probability: f64) -> Signal {
        let pred = probability - 0.5; // Center around 0

        if probability >= 0.7 {
            Signal::StrongBuy
        } else if probability >= 0.55 {
            Signal::Buy
        } else if probability <= 0.3 {
            Signal::StrongSell
        } else if probability <= 0.45 {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    /// Generate signals for a batch of predictions
    pub fn generate_batch(&self, predictions: &[f64], confidences: Option<&[f64]>) -> Vec<Signal> {
        predictions
            .iter()
            .enumerate()
            .map(|(i, &pred)| {
                let conf = confidences.map_or(1.0, |c| c[i]);
                self.generate(pred, conf)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_direction() {
        assert_eq!(Signal::StrongBuy.to_direction(), 1.0);
        assert_eq!(Signal::Hold.to_direction(), 0.0);
        assert_eq!(Signal::StrongSell.to_direction(), -1.0);
    }

    #[test]
    fn test_signal_generator() {
        let generator = SignalGenerator::new(SignalConfig::default());

        assert_eq!(generator.generate(0.01, 0.8), Signal::StrongBuy);
        assert_eq!(generator.generate(0.002, 0.8), Signal::Buy);
        assert_eq!(generator.generate(0.0, 0.8), Signal::Hold);
        assert_eq!(generator.generate(-0.002, 0.8), Signal::Sell);
        assert_eq!(generator.generate(-0.01, 0.8), Signal::StrongSell);
    }

    #[test]
    fn test_low_confidence() {
        let generator = SignalGenerator::new(SignalConfig::default());

        // Low confidence should return Hold
        assert_eq!(generator.generate(0.01, 0.3), Signal::Hold);
    }
}
