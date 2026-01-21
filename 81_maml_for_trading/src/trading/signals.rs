//! Trading signal types and utilities.

use serde::{Deserialize, Serialize};

/// Trading signal indicating recommended action
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradingSignal {
    /// Strong buy signal
    StrongBuy,
    /// Buy signal
    Buy,
    /// Hold/neutral signal
    Hold,
    /// Sell signal
    Sell,
    /// Strong sell signal
    StrongSell,
}

impl TradingSignal {
    /// Create signal from prediction value
    ///
    /// # Arguments
    /// * `prediction` - Model prediction (expected return)
    /// * `threshold` - Threshold for buy/sell signals
    /// * `strong_threshold` - Threshold for strong buy/sell signals
    pub fn from_prediction(prediction: f64, threshold: f64, strong_threshold: f64) -> Self {
        if prediction > strong_threshold {
            TradingSignal::StrongBuy
        } else if prediction > threshold {
            TradingSignal::Buy
        } else if prediction < -strong_threshold {
            TradingSignal::StrongSell
        } else if prediction < -threshold {
            TradingSignal::Sell
        } else {
            TradingSignal::Hold
        }
    }

    /// Convert signal to position (-1, 0, or 1)
    pub fn to_position(&self) -> i32 {
        match self {
            TradingSignal::StrongBuy | TradingSignal::Buy => 1,
            TradingSignal::Hold => 0,
            TradingSignal::Sell | TradingSignal::StrongSell => -1,
        }
    }

    /// Convert signal to position size (0.0 to 1.0)
    pub fn to_position_size(&self) -> f64 {
        match self {
            TradingSignal::StrongBuy => 1.0,
            TradingSignal::Buy => 0.5,
            TradingSignal::Hold => 0.0,
            TradingSignal::Sell => -0.5,
            TradingSignal::StrongSell => -1.0,
        }
    }

    /// Check if signal is bullish
    pub fn is_bullish(&self) -> bool {
        matches!(self, TradingSignal::StrongBuy | TradingSignal::Buy)
    }

    /// Check if signal is bearish
    pub fn is_bearish(&self) -> bool {
        matches!(self, TradingSignal::Sell | TradingSignal::StrongSell)
    }

    /// Check if signal is neutral
    pub fn is_neutral(&self) -> bool {
        matches!(self, TradingSignal::Hold)
    }
}

impl std::fmt::Display for TradingSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradingSignal::StrongBuy => write!(f, "STRONG BUY"),
            TradingSignal::Buy => write!(f, "BUY"),
            TradingSignal::Hold => write!(f, "HOLD"),
            TradingSignal::Sell => write!(f, "SELL"),
            TradingSignal::StrongSell => write!(f, "STRONG SELL"),
        }
    }
}

/// Signal with confidence and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedSignal {
    /// The trading signal
    pub signal: TradingSignal,
    /// Raw prediction value
    pub prediction: f64,
    /// Confidence level (0-1)
    pub confidence: f64,
    /// Timestamp
    pub timestamp: i64,
    /// Symbol
    pub symbol: String,
}

impl DetailedSignal {
    /// Create a new detailed signal
    pub fn new(
        signal: TradingSignal,
        prediction: f64,
        confidence: f64,
        timestamp: i64,
        symbol: String,
    ) -> Self {
        Self {
            signal,
            prediction,
            confidence,
            timestamp,
            symbol,
        }
    }

    /// Create from prediction with automatic signal generation
    pub fn from_prediction(
        prediction: f64,
        threshold: f64,
        strong_threshold: f64,
        timestamp: i64,
        symbol: String,
    ) -> Self {
        let signal = TradingSignal::from_prediction(prediction, threshold, strong_threshold);
        let confidence = (prediction.abs() / strong_threshold).min(1.0);

        Self {
            signal,
            prediction,
            confidence,
            timestamp,
            symbol,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_from_prediction() {
        let threshold = 0.001;
        let strong_threshold = 0.005;

        assert_eq!(
            TradingSignal::from_prediction(0.01, threshold, strong_threshold),
            TradingSignal::StrongBuy
        );
        assert_eq!(
            TradingSignal::from_prediction(0.002, threshold, strong_threshold),
            TradingSignal::Buy
        );
        assert_eq!(
            TradingSignal::from_prediction(0.0005, threshold, strong_threshold),
            TradingSignal::Hold
        );
        assert_eq!(
            TradingSignal::from_prediction(-0.002, threshold, strong_threshold),
            TradingSignal::Sell
        );
        assert_eq!(
            TradingSignal::from_prediction(-0.01, threshold, strong_threshold),
            TradingSignal::StrongSell
        );
    }

    #[test]
    fn test_signal_to_position() {
        assert_eq!(TradingSignal::StrongBuy.to_position(), 1);
        assert_eq!(TradingSignal::Buy.to_position(), 1);
        assert_eq!(TradingSignal::Hold.to_position(), 0);
        assert_eq!(TradingSignal::Sell.to_position(), -1);
        assert_eq!(TradingSignal::StrongSell.to_position(), -1);
    }

    #[test]
    fn test_signal_checks() {
        assert!(TradingSignal::StrongBuy.is_bullish());
        assert!(TradingSignal::Buy.is_bullish());
        assert!(!TradingSignal::Hold.is_bullish());

        assert!(TradingSignal::Sell.is_bearish());
        assert!(TradingSignal::StrongSell.is_bearish());
        assert!(!TradingSignal::Buy.is_bearish());

        assert!(TradingSignal::Hold.is_neutral());
        assert!(!TradingSignal::Buy.is_neutral());
    }
}
