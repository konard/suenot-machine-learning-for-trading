//! Candlestick (OHLCV) data model.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// OHLCV candlestick data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Timestamp of the candle open
    pub timestamp: DateTime<Utc>,
    /// Symbol (e.g., "BTCUSDT")
    pub symbol: String,
    /// Open price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Close price
    pub close: f64,
    /// Volume
    pub volume: f64,
    /// Turnover (volume * price)
    pub turnover: f64,
}

impl Candle {
    /// Create a new candle.
    pub fn new(
        timestamp: DateTime<Utc>,
        symbol: String,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        let turnover = volume * (open + high + low + close) / 4.0;
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

    /// Calculate the typical price (HLC/3).
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate the range (high - low).
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Check if the candle is bullish (close > open).
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if the candle is bearish (close < open).
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Calculate the body size as a percentage of the range.
    pub fn body_ratio(&self) -> f64 {
        let range = self.range();
        if range == 0.0 {
            return 0.0;
        }
        (self.close - self.open).abs() / range
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_bullish() {
        let candle = Candle::new(
            Utc::now(),
            "BTCUSDT".to_string(),
            100.0,
            110.0,
            95.0,
            105.0,
            1000.0,
        );
        assert!(candle.is_bullish());
        assert!(!candle.is_bearish());
    }

    #[test]
    fn test_typical_price() {
        let candle = Candle::new(
            Utc::now(),
            "BTCUSDT".to_string(),
            100.0,
            120.0,
            90.0,
            110.0,
            1000.0,
        );
        assert!((candle.typical_price() - 106.666).abs() < 0.01);
    }
}
