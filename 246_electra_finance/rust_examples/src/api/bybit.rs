//! Bybit API client for fetching cryptocurrency market data
//!
//! This module provides a client for interacting with Bybit's public API
//! to fetch OHLCV (Open, High, Low, Close, Volume) data.

use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur when interacting with the Bybit API
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("Failed to parse response: {0}")]
    ParseError(#[from] serde_json::Error),

    #[error("API returned error: {code} - {message}")]
    ApiError { code: i32, message: String },

    #[error("Invalid interval: {0}")]
    InvalidInterval(String),
}

/// Kline/candlestick interval
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interval {
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

impl Interval {
    /// Convert interval to API string
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

    /// Get interval duration in minutes
    pub fn minutes(&self) -> u64 {
        match self {
            Interval::Min1 => 1,
            Interval::Min3 => 3,
            Interval::Min5 => 5,
            Interval::Min15 => 15,
            Interval::Min30 => 30,
            Interval::Hour1 => 60,
            Interval::Hour2 => 120,
            Interval::Hour4 => 240,
            Interval::Hour6 => 360,
            Interval::Hour12 => 720,
            Interval::Day1 => 1440,
            Interval::Week1 => 10080,
            Interval::Month1 => 43200,
        }
    }
}

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub turnover: f64,
}

impl Kline {
    /// Get the datetime for this kline
    pub fn datetime(&self) -> DateTime<Utc> {
        Utc.timestamp_millis_opt(self.timestamp).unwrap()
    }

    /// Compute the return from open to close
    pub fn return_pct(&self) -> f64 {
        if self.open == 0.0 {
            return 0.0;
        }
        (self.close - self.open) / self.open
    }
}

/// Bybit API response structure
#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitResult,
}

/// Bybit API result structure
#[derive(Debug, Deserialize)]
struct BybitResult {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

/// Client for the Bybit API
pub struct BybitClient {
    base_url: String,
    client: reqwest::blocking::Client,
}

impl BybitClient {
    /// Create a new Bybit API client
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Create a client with a custom base URL (for testing)
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Fetch kline/candlestick data for a trading pair
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Candlestick interval
    /// * `limit` - Number of candles to fetch (max 200)
    /// * `start_time` - Optional start timestamp in milliseconds
    /// * `end_time` - Optional end timestamp in milliseconds
    pub fn get_klines(
        &self,
        symbol: &str,
        interval: Interval,
        limit: u32,
        start_time: Option<i64>,
        end_time: Option<i64>,
    ) -> Result<Vec<Kline>, BybitError> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let mut params = vec![
            ("category", "spot".to_string()),
            ("symbol", symbol.to_string()),
            ("interval", interval.as_str().to_string()),
            ("limit", limit.min(200).to_string()),
        ];

        if let Some(start) = start_time {
            params.push(("start", start.to_string()));
        }
        if let Some(end) = end_time {
            params.push(("end", end.to_string()));
        }

        let response: BybitResponse = self
            .client
            .get(&url)
            .query(&params)
            .send()?
            .json()?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let klines: Vec<Kline> = response
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() >= 7 {
                    Some(Kline {
                        timestamp: row[0].parse().ok()?,
                        open: row[1].parse().ok()?,
                        high: row[2].parse().ok()?,
                        low: row[3].parse().ok()?,
                        close: row[4].parse().ok()?,
                        volume: row[5].parse().ok()?,
                        turnover: row[6].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(klines)
    }

    /// Fetch historical klines across multiple API calls
    ///
    /// The Bybit API limits responses to 200 candles per call.
    /// This method fetches data in batches to cover a larger time range.
    pub fn get_historical_klines(
        &self,
        symbol: &str,
        interval: Interval,
        start_time: i64,
        end_time: i64,
    ) -> Result<Vec<Kline>, BybitError> {
        let mut all_klines = Vec::new();
        let mut current_end = end_time;
        let interval_ms = interval.minutes() as i64 * 60 * 1000;

        loop {
            let klines = self.get_klines(
                symbol,
                interval,
                200,
                Some(start_time),
                Some(current_end),
            )?;

            if klines.is_empty() {
                break;
            }

            let min_timestamp = klines.iter().map(|k| k.timestamp).min().unwrap();
            all_klines.extend(klines);

            if min_timestamp <= start_time {
                break;
            }

            current_end = min_timestamp - interval_ms;
        }

        all_klines.sort_by_key(|k| k.timestamp);
        all_klines.dedup_by_key(|k| k.timestamp);

        Ok(all_klines)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}
