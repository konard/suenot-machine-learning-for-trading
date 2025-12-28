//! Candlestick (OHLCV) data structure

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Timestamp in milliseconds
    pub timestamp: i64,
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
}

impl Candle {
    /// Create a new candle
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

    /// Get datetime from timestamp
    pub fn datetime(&self) -> Option<DateTime<Utc>> {
        DateTime::from_timestamp_millis(self.timestamp)
    }

    /// Calculate return from open to close
    pub fn return_pct(&self) -> f64 {
        if self.open != 0.0 {
            (self.close - self.open) / self.open
        } else {
            0.0
        }
    }

    /// Calculate candle body size (absolute)
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Calculate upper shadow (wick)
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.close.max(self.open)
    }

    /// Calculate lower shadow (wick)
    pub fn lower_shadow(&self) -> f64 {
        self.close.min(self.open) - self.low
    }

    /// Calculate range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Check if bullish (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if bearish (close < open)
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Typical price (high + low + close) / 3
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// VWAP-like measure for single candle
    pub fn vwap(&self) -> f64 {
        self.typical_price()
    }
}

/// Calculate returns from a series of candles
pub fn calculate_returns(candles: &[Candle]) -> Vec<f64> {
    if candles.len() < 2 {
        return vec![];
    }

    candles
        .windows(2)
        .map(|w| {
            if w[0].close != 0.0 {
                (w[1].close - w[0].close) / w[0].close
            } else {
                0.0
            }
        })
        .collect()
}

/// Calculate log returns from a series of candles
pub fn calculate_log_returns(candles: &[Candle]) -> Vec<f64> {
    if candles.len() < 2 {
        return vec![];
    }

    candles
        .windows(2)
        .map(|w| {
            if w[0].close > 0.0 && w[1].close > 0.0 {
                (w[1].close / w[0].close).ln()
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_calculations() {
        let candle = Candle::new(1000, 100.0, 110.0, 95.0, 105.0, 1000.0);

        assert!(candle.is_bullish());
        assert!(!candle.is_bearish());
        assert_eq!(candle.body(), 5.0);
        assert_eq!(candle.range(), 15.0);
        assert_eq!(candle.upper_shadow(), 5.0);
        assert_eq!(candle.lower_shadow(), 5.0);
        assert!((candle.return_pct() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_returns() {
        let candles = vec![
            Candle::new(1, 100.0, 100.0, 100.0, 100.0, 100.0),
            Candle::new(2, 100.0, 100.0, 100.0, 110.0, 100.0),
            Candle::new(3, 110.0, 110.0, 110.0, 105.0, 100.0),
        ];

        let returns = calculate_returns(&candles);
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.10).abs() < 1e-10);
        assert!((returns[1] - (-0.0454545)).abs() < 1e-5);
    }
}
