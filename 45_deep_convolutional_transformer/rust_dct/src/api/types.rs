//! Bybit API response types

use serde::{Deserialize, Serialize};

/// Kline (candlestick) data from Bybit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KlineData {
    /// Start time in milliseconds
    pub timestamp: i64,
    /// Open price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Close price
    pub close: f64,
    /// Trading volume
    pub volume: f64,
    /// Turnover (quote volume)
    pub turnover: f64,
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

/// Kline result from API
#[derive(Debug, Deserialize)]
pub struct KlineResult {
    pub category: String,
    pub symbol: String,
    pub list: Vec<Vec<String>>,
}

/// Full kline response
pub type KlineResponse = ApiResponse<KlineResult>;

impl KlineData {
    /// Parse kline from API response list
    pub fn from_list(list: &[String]) -> Option<Self> {
        if list.len() < 7 {
            return None;
        }

        Some(KlineData {
            timestamp: list[0].parse().ok()?,
            open: list[1].parse().ok()?,
            high: list[2].parse().ok()?,
            low: list[3].parse().ok()?,
            close: list[4].parse().ok()?,
            volume: list[5].parse().ok()?,
            turnover: list[6].parse().ok()?,
        })
    }
}
