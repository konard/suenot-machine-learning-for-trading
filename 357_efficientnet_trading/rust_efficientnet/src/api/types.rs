//! Type definitions for Bybit API responses

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Kline/Candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub turnover: f64,
}

impl Kline {
    /// Create a new Kline
    pub fn new(
        timestamp: DateTime<Utc>,
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

    /// Calculate the range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Check if bullish (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Calculate body size
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }
}

/// Collection of klines with utility methods
#[derive(Debug, Clone, Default)]
pub struct KlineData {
    pub klines: Vec<Kline>,
    pub symbol: String,
    pub interval: String,
}

impl KlineData {
    /// Create new KlineData
    pub fn new(symbol: String, interval: String, klines: Vec<Kline>) -> Self {
        Self {
            klines,
            symbol,
            interval,
        }
    }

    /// Get close prices as a vector
    pub fn close_prices(&self) -> Vec<f64> {
        self.klines.iter().map(|k| k.close).collect()
    }

    /// Get open prices as a vector
    pub fn open_prices(&self) -> Vec<f64> {
        self.klines.iter().map(|k| k.open).collect()
    }

    /// Get high prices as a vector
    pub fn high_prices(&self) -> Vec<f64> {
        self.klines.iter().map(|k| k.high).collect()
    }

    /// Get low prices as a vector
    pub fn low_prices(&self) -> Vec<f64> {
        self.klines.iter().map(|k| k.low).collect()
    }

    /// Get volumes as a vector
    pub fn volumes(&self) -> Vec<f64> {
        self.klines.iter().map(|k| k.volume).collect()
    }

    /// Get the number of klines
    pub fn len(&self) -> usize {
        self.klines.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.klines.is_empty()
    }

    /// Get latest kline
    pub fn latest(&self) -> Option<&Kline> {
        self.klines.last()
    }

    /// Calculate returns
    pub fn returns(&self) -> Vec<f64> {
        let closes = self.close_prices();
        closes
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }
}

/// Ticker information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
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

impl Ticker {
    /// Get last price as f64
    pub fn last_price_f64(&self) -> f64 {
        self.last_price.parse().unwrap_or(0.0)
    }

    /// Get 24h price change percentage as f64
    pub fn price_change_pct(&self) -> f64 {
        self.price_24h_pcnt.parse::<f64>().unwrap_or(0.0) * 100.0
    }
}

/// API response wrapper
#[derive(Debug, Deserialize)]
pub struct ApiResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
}

/// Klines result from API
#[derive(Debug, Deserialize)]
pub struct KlinesResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// Tickers result from API
#[derive(Debug, Deserialize)]
pub struct TickersResult {
    pub category: String,
    pub list: Vec<Ticker>,
}
