//! API types for Bybit responses

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Unix timestamp in milliseconds
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
    /// Turnover (quote volume)
    pub turnover: f64,
}

impl Candle {
    /// Create a new candle from raw values
    pub fn new(
        timestamp: i64,
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

    /// Convert timestamp to DateTime
    pub fn datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(self.timestamp)
            .unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap())
    }

    /// Calculate the body ratio (|close - open| / (high - low))
    pub fn body_ratio(&self) -> f64 {
        let range = self.high - self.low;
        if range > 0.0 {
            (self.close - self.open).abs() / range
        } else {
            0.0
        }
    }

    /// Check if candle is bullish (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Calculate the return from open to close
    pub fn return_pct(&self) -> f64 {
        if self.open > 0.0 {
            (self.close - self.open) / self.open
        } else {
            0.0
        }
    }
}

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
pub struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
    pub time: i64,
}

/// Kline (candlestick) response from Bybit
#[derive(Debug, Deserialize)]
pub struct KlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// Type alias for kline API response
pub type KlineResponse = BybitResponse<KlineResult>;

/// Ticker information
#[derive(Debug, Deserialize)]
pub struct TickerInfo {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "highPrice24h")]
    pub high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_price_24h: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    #[serde(rename = "turnover24h")]
    pub turnover_24h: String,
    #[serde(rename = "price24hPcnt")]
    pub price_24h_pcnt: String,
}

/// Ticker response result
#[derive(Debug, Deserialize)]
pub struct TickerResult {
    pub category: String,
    pub list: Vec<TickerInfo>,
}

/// Type alias for ticker API response
pub type TickerResponse = BybitResponse<TickerResult>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_creation() {
        let candle = Candle::new(
            1704067200000, // 2024-01-01 00:00:00 UTC
            50000.0,
            50500.0,
            49800.0,
            50300.0,
            1000.0,
            50000000.0,
        );

        assert!(candle.is_bullish());
        assert!((candle.body_ratio() - 0.4285714).abs() < 0.001);
    }

    #[test]
    fn test_candle_return() {
        let candle = Candle::new(0, 100.0, 110.0, 95.0, 105.0, 1000.0, 100000.0);
        assert!((candle.return_pct() - 0.05).abs() < 0.001);
    }
}
