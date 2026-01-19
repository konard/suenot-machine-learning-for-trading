//! Data Types
//!
//! Common data structures for market data.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// OHLCV candle data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCV {
    /// Timestamp
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
    /// Turnover (optional)
    pub turnover: Option<f64>,
}

/// Single candle with DateTime timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
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
    /// Turnover
    pub turnover: Option<f64>,
}

impl Candle {
    /// Create new candle
    pub fn new(
        timestamp: DateTime<Utc>,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            turnover: None,
        }
    }

    /// Typical price (HLC/3)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Is bullish candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Candle range
    pub fn range(&self) -> f64 {
        self.high - self.low
    }
}

/// Price series (collection of candles)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceSeries {
    /// Symbol
    pub symbol: String,
    /// Interval
    pub interval: String,
    /// Candles
    pub candles: Vec<Candle>,
}

impl PriceSeries {
    /// Create new empty series
    pub fn new(symbol: String, interval: String) -> Self {
        Self {
            symbol,
            interval,
            candles: Vec::new(),
        }
    }

    /// Add candle
    pub fn push(&mut self, candle: Candle) {
        self.candles.push(candle);
    }

    /// Get closing prices
    pub fn closes(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.close).collect()
    }

    /// Get volumes
    pub fn volumes(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.volume).collect()
    }

    /// Number of candles
    pub fn len(&self) -> usize {
        self.candles.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.candles.is_empty()
    }

    /// First candle
    pub fn first(&self) -> Option<&Candle> {
        self.candles.first()
    }

    /// Last candle
    pub fn last(&self) -> Option<&Candle> {
        self.candles.last()
    }

    /// Sort by time
    pub fn sort_by_time(&mut self) {
        self.candles.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    }

    /// Calculate simple returns
    pub fn returns(&self) -> Vec<f64> {
        let closes = self.closes();
        if closes.len() < 2 {
            return Vec::new();
        }
        closes.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle() {
        let candle = Candle::new(Utc::now(), 100.0, 110.0, 95.0, 105.0, 1000.0);
        assert!(candle.is_bullish());
        assert_eq!(candle.range(), 15.0);
    }

    #[test]
    fn test_price_series() {
        let mut series = PriceSeries::new("BTCUSDT".to_string(), "1d".to_string());
        series.push(Candle::new(Utc::now(), 100.0, 110.0, 95.0, 100.0, 1000.0));
        series.push(Candle::new(Utc::now(), 100.0, 120.0, 100.0, 110.0, 1000.0));

        assert_eq!(series.len(), 2);
        let returns = series.returns();
        assert_eq!(returns.len(), 1);
        assert!((returns[0] - 0.1).abs() < 0.001);
    }
}
