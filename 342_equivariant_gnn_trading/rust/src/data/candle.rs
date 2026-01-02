//! Candlestick (OHLCV) Data Structure
//!
//! Represents a single candlestick with open, high, low, close prices and volume.

use serde::{Deserialize, Serialize};

/// Represents a single OHLCV candlestick
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Candle {
    /// Timestamp in milliseconds
    pub timestamp: u64,

    /// Opening price
    pub open: f64,

    /// Highest price during the period
    pub high: f64,

    /// Lowest price during the period
    pub low: f64,

    /// Closing price
    pub close: f64,

    /// Trading volume
    pub volume: f64,

    /// Turnover (quote volume)
    pub turnover: f64,
}

impl Candle {
    /// Create a new candle
    pub fn new(
        timestamp: u64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        turnover: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            turnover,
        }
    }

    /// Calculate the return from open to close
    pub fn return_pct(&self) -> f64 {
        (self.close - self.open) / self.open
    }

    /// Calculate log return
    pub fn log_return(&self) -> f64 {
        (self.close / self.open).ln()
    }

    /// Calculate the high-low range as percentage of close
    pub fn range_pct(&self) -> f64 {
        (self.high - self.low) / self.close
    }

    /// Calculate body size (absolute difference between open and close)
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Check if candle is bullish (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if candle is bearish (close < open)
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Calculate upper shadow
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.close.max(self.open)
    }

    /// Calculate lower shadow
    pub fn lower_shadow(&self) -> f64 {
        self.close.min(self.open) - self.low
    }

    /// Calculate close position within the range (0 = low, 1 = high)
    pub fn close_position(&self) -> f64 {
        if (self.high - self.low).abs() < 1e-10 {
            0.5
        } else {
            (self.close - self.low) / (self.high - self.low)
        }
    }

    /// Calculate typical price (average of high, low, close)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate VWAP-like weighted price
    pub fn vwap_price(&self) -> f64 {
        if self.volume.abs() < 1e-10 {
            self.typical_price()
        } else {
            self.turnover / self.volume
        }
    }
}

/// Calculate returns from a series of candles
pub fn calculate_returns(candles: &[Candle]) -> Vec<f64> {
    if candles.len() < 2 {
        return vec![];
    }

    candles
        .windows(2)
        .map(|w| (w[1].close - w[0].close) / w[0].close)
        .collect()
}

/// Calculate log returns from a series of candles
pub fn calculate_log_returns(candles: &[Candle]) -> Vec<f64> {
    if candles.len() < 2 {
        return vec![];
    }

    candles
        .windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_creation() {
        let candle = Candle::new(
            1700000000000,
            100.0,
            105.0,
            98.0,
            103.0,
            1000.0,
            100000.0,
        );

        assert_eq!(candle.timestamp, 1700000000000);
        assert!((candle.return_pct() - 0.03).abs() < 1e-10);
    }

    #[test]
    fn test_bullish_bearish() {
        let bullish = Candle::new(0, 100.0, 110.0, 99.0, 108.0, 100.0, 10000.0);
        let bearish = Candle::new(0, 100.0, 105.0, 95.0, 97.0, 100.0, 9700.0);

        assert!(bullish.is_bullish());
        assert!(!bullish.is_bearish());
        assert!(!bearish.is_bullish());
        assert!(bearish.is_bearish());
    }

    #[test]
    fn test_calculate_returns() {
        let candles = vec![
            Candle::new(0, 100.0, 100.0, 100.0, 100.0, 100.0, 10000.0),
            Candle::new(1, 100.0, 110.0, 100.0, 110.0, 100.0, 11000.0),
            Candle::new(2, 110.0, 115.0, 105.0, 105.0, 100.0, 10500.0),
        ];

        let returns = calculate_returns(&candles);
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.1).abs() < 1e-10);
        assert!((returns[1] - (-0.0454545)).abs() < 1e-5);
    }

    #[test]
    fn test_close_position() {
        let candle = Candle::new(0, 100.0, 110.0, 90.0, 100.0, 100.0, 10000.0);
        assert!((candle.close_position() - 0.5).abs() < 1e-10);

        let high_close = Candle::new(0, 90.0, 110.0, 90.0, 110.0, 100.0, 11000.0);
        assert!((high_close.close_position() - 1.0).abs() < 1e-10);
    }
}
