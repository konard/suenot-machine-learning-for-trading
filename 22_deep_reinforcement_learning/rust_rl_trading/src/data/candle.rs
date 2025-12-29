//! Candlestick (OHLCV) data structure.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Represents a single candlestick (OHLCV) data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Opening timestamp
    pub timestamp: DateTime<Utc>,
    /// Symbol (e.g., "BTCUSDT")
    pub symbol: String,
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
    /// Turnover (volume * price)
    pub turnover: f64,
}

impl Candle {
    /// Create a new Candle
    pub fn new(
        timestamp: DateTime<Utc>,
        symbol: String,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        turnover: f64,
    ) -> Self {
        Self {
            timestamp,
            symbol,
            open,
            high,
            low,
            close,
            volume,
            turnover,
        }
    }

    /// Calculate the return (percentage change from open to close)
    pub fn returns(&self) -> f64 {
        if self.open > 0.0 {
            (self.close - self.open) / self.open
        } else {
            0.0
        }
    }

    /// Calculate the body size (absolute difference between open and close)
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Calculate the upper wick size
    pub fn upper_wick(&self) -> f64 {
        self.high - self.open.max(self.close)
    }

    /// Calculate the lower wick size
    pub fn lower_wick(&self) -> f64 {
        self.open.min(self.close) - self.low
    }

    /// Check if it's a bullish candle (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if it's a bearish candle (close < open)
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Calculate the range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_returns() {
        let candle = Candle::new(
            Utc::now(),
            "BTCUSDT".to_string(),
            100.0,
            110.0,
            95.0,
            105.0,
            1000.0,
            100000.0,
        );

        assert!((candle.returns() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_bullish_bearish() {
        let bullish = Candle::new(
            Utc::now(),
            "BTCUSDT".to_string(),
            100.0,
            110.0,
            95.0,
            105.0,
            1000.0,
            100000.0,
        );

        let bearish = Candle::new(
            Utc::now(),
            "BTCUSDT".to_string(),
            105.0,
            110.0,
            95.0,
            100.0,
            1000.0,
            100000.0,
        );

        assert!(bullish.is_bullish());
        assert!(!bullish.is_bearish());
        assert!(bearish.is_bearish());
        assert!(!bearish.is_bullish());
    }
}
