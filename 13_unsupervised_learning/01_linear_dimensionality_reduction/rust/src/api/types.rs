//! Type definitions for Bybit API responses

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Bybit API response wrapper
#[derive(Debug, Clone, Deserialize)]
pub struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
}

/// Kline (candlestick) data result
#[derive(Debug, Clone, Deserialize)]
pub struct KlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<KlineData>,
}

/// Individual kline data point
/// Format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
#[derive(Debug, Clone, Deserialize)]
pub struct KlineData(
    pub String, // start time
    pub String, // open
    pub String, // high
    pub String, // low
    pub String, // close
    pub String, // volume
    pub String, // turnover
);

impl KlineData {
    pub fn timestamp(&self) -> i64 {
        self.0.parse().unwrap_or(0)
    }

    pub fn open(&self) -> f64 {
        self.1.parse().unwrap_or(0.0)
    }

    pub fn high(&self) -> f64 {
        self.2.parse().unwrap_or(0.0)
    }

    pub fn low(&self) -> f64 {
        self.3.parse().unwrap_or(0.0)
    }

    pub fn close(&self) -> f64 {
        self.4.parse().unwrap_or(0.0)
    }

    pub fn volume(&self) -> f64 {
        self.5.parse().unwrap_or(0.0)
    }

    pub fn turnover(&self) -> f64 {
        self.6.parse().unwrap_or(0.0)
    }
}

/// OHLCV data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCV {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl From<KlineData> for OHLCV {
    fn from(k: KlineData) -> Self {
        Self {
            timestamp: k.timestamp(),
            open: k.open(),
            high: k.high(),
            low: k.low(),
            close: k.close(),
            volume: k.volume(),
        }
    }
}

/// Instruments info result
#[derive(Debug, Clone, Deserialize)]
pub struct InstrumentsResult {
    pub category: String,
    pub list: Vec<InstrumentInfo>,
}

/// Individual instrument information
#[derive(Debug, Clone, Deserialize)]
pub struct InstrumentInfo {
    pub symbol: String,
    #[serde(rename = "baseCoin")]
    pub base_coin: String,
    #[serde(rename = "quoteCoin")]
    pub quote_coin: String,
    pub status: String,
}

/// Ticker result
#[derive(Debug, Clone, Deserialize)]
pub struct TickersResult {
    pub category: String,
    pub list: Vec<TickerInfo>,
}

/// Individual ticker information
#[derive(Debug, Clone, Deserialize)]
pub struct TickerInfo {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    #[serde(rename = "turnover24h")]
    pub turnover_24h: String,
}

impl TickerInfo {
    pub fn volume_24h_f64(&self) -> f64 {
        self.volume_24h.parse().unwrap_or(0.0)
    }

    pub fn turnover_24h_f64(&self) -> f64 {
        self.turnover_24h.parse().unwrap_or(0.0)
    }
}

/// Timeframe enum for kline data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Timeframe {
    Min1,
    Min3,
    Min5,
    Min15,
    Min30,
    Hour1,
    Hour2,
    Hour4,
    Hour6,
    Hour12,
    Day1,
    Week1,
    Month1,
}

impl Timeframe {
    pub fn as_str(&self) -> &'static str {
        match self {
            Timeframe::Min1 => "1",
            Timeframe::Min3 => "3",
            Timeframe::Min5 => "5",
            Timeframe::Min15 => "15",
            Timeframe::Min30 => "30",
            Timeframe::Hour1 => "60",
            Timeframe::Hour2 => "120",
            Timeframe::Hour4 => "240",
            Timeframe::Hour6 => "360",
            Timeframe::Hour12 => "720",
            Timeframe::Day1 => "D",
            Timeframe::Week1 => "W",
            Timeframe::Month1 => "M",
        }
    }

    pub fn to_minutes(&self) -> i64 {
        match self {
            Timeframe::Min1 => 1,
            Timeframe::Min3 => 3,
            Timeframe::Min5 => 5,
            Timeframe::Min15 => 15,
            Timeframe::Min30 => 30,
            Timeframe::Hour1 => 60,
            Timeframe::Hour2 => 120,
            Timeframe::Hour4 => 240,
            Timeframe::Hour6 => 360,
            Timeframe::Hour12 => 720,
            Timeframe::Day1 => 1440,
            Timeframe::Week1 => 10080,
            Timeframe::Month1 => 43200,
        }
    }
}
