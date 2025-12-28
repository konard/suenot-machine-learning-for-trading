//! Type definitions for Bybit API responses

use serde::{Deserialize, Serialize};

/// Trading pair symbol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol(pub String);

impl Symbol {
    pub fn new(s: &str) -> Self {
        Self(s.to_string())
    }

    pub fn btcusdt() -> Self {
        Self("BTCUSDT".to_string())
    }

    pub fn ethusdt() -> Self {
        Self("ETHUSDT".to_string())
    }

    pub fn solusdt() -> Self {
        Self("SOLUSDT".to_string())
    }
}

impl AsRef<str> for Symbol {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Kline interval
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Interval {
    #[serde(rename = "1")]
    Min1,
    #[serde(rename = "3")]
    Min3,
    #[serde(rename = "5")]
    Min5,
    #[serde(rename = "15")]
    Min15,
    #[serde(rename = "30")]
    Min30,
    #[serde(rename = "60")]
    Hour1,
    #[serde(rename = "120")]
    Hour2,
    #[serde(rename = "240")]
    Hour4,
    #[serde(rename = "360")]
    Hour6,
    #[serde(rename = "720")]
    Hour12,
    #[serde(rename = "D")]
    Day1,
    #[serde(rename = "W")]
    Week1,
    #[serde(rename = "M")]
    Month1,
}

impl Interval {
    pub fn as_str(&self) -> &'static str {
        match self {
            Interval::Min1 => "1",
            Interval::Min3 => "3",
            Interval::Min5 => "5",
            Interval::Min15 => "15",
            Interval::Min30 => "30",
            Interval::Hour1 => "60",
            Interval::Hour2 => "120",
            Interval::Hour4 => "240",
            Interval::Hour6 => "360",
            Interval::Hour12 => "720",
            Interval::Day1 => "D",
            Interval::Week1 => "W",
            Interval::Month1 => "M",
        }
    }

    /// Returns interval duration in milliseconds
    pub fn duration_ms(&self) -> i64 {
        match self {
            Interval::Min1 => 60_000,
            Interval::Min3 => 180_000,
            Interval::Min5 => 300_000,
            Interval::Min15 => 900_000,
            Interval::Min30 => 1_800_000,
            Interval::Hour1 => 3_600_000,
            Interval::Hour2 => 7_200_000,
            Interval::Hour4 => 14_400_000,
            Interval::Hour6 => 21_600_000,
            Interval::Hour12 => 43_200_000,
            Interval::Day1 => 86_400_000,
            Interval::Week1 => 604_800_000,
            Interval::Month1 => 2_592_000_000,
        }
    }
}

/// Raw kline response from Bybit API
#[derive(Debug, Deserialize)]
pub struct KlineResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: KlineResult,
}

#[derive(Debug, Deserialize)]
pub struct KlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<KlineData>,
}

/// Single kline data point
/// Format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
#[derive(Debug, Deserialize)]
pub struct KlineData(
    pub String, // startTime
    pub String, // openPrice
    pub String, // highPrice
    pub String, // lowPrice
    pub String, // closePrice
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
