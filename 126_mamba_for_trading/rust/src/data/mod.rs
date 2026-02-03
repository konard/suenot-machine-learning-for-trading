//! Data processing and feature engineering module.

pub mod features;
pub mod loader;

use serde::{Deserialize, Serialize};

/// OHLCV bar representing a single candlestick.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OhlcvBar {
    /// Timestamp in milliseconds
    pub timestamp: i64,
    /// Opening price
    pub open: f64,
    /// Highest price
    pub high: f64,
    /// Lowest price
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Trading volume
    pub volume: f64,
}

impl OhlcvBar {
    /// Create a new OHLCV bar.
    pub fn new(timestamp: i64, open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Calculate the true range.
    pub fn true_range(&self, prev_close: f64) -> f64 {
        let hl = self.high - self.low;
        let hc = (self.high - prev_close).abs();
        let lc = (self.low - prev_close).abs();
        hl.max(hc).max(lc)
    }

    /// Calculate the body of the candle.
    pub fn body(&self) -> f64 {
        self.close - self.open
    }

    /// Check if the candle is bullish (green).
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Calculate returns from previous bar.
    pub fn returns(&self, prev_close: f64) -> f64 {
        (self.close - prev_close) / prev_close
    }
}

/// Trading signal enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradingSignal {
    Buy,
    Hold,
    Sell,
}

impl TradingSignal {
    /// Convert from class index (0=Sell, 1=Hold, 2=Buy).
    pub fn from_class(class: usize) -> Self {
        match class {
            0 => TradingSignal::Sell,
            1 => TradingSignal::Hold,
            2 => TradingSignal::Buy,
            _ => TradingSignal::Hold,
        }
    }

    /// Convert to class index.
    pub fn to_class(&self) -> usize {
        match self {
            TradingSignal::Sell => 0,
            TradingSignal::Hold => 1,
            TradingSignal::Buy => 2,
        }
    }
}

/// Prediction result with confidence scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub signal: TradingSignal,
    pub confidence: f64,
    pub probabilities: [f64; 3], // [sell, hold, buy]
}

impl Prediction {
    /// Create a new prediction from probabilities.
    pub fn from_probabilities(probs: [f64; 3]) -> Self {
        let (max_idx, &max_prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        Self {
            signal: TradingSignal::from_class(max_idx),
            confidence: max_prob,
            probabilities: probs,
        }
    }
}
