//! # Bybit API Client
//!
//! Client for fetching market data from Bybit exchange.
//!
//! ## Example
//!
//! ```rust,no_run
//! use score_matching_trading::api::BybitClient;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let client = BybitClient::new();
//!     let klines = client.get_klines("BTCUSDT", "1h", 100).await?;
//!     println!("Fetched {} candles", klines.len());
//!     Ok(())
//! }
//! ```

use crate::utils::Candle;
use chrono::{DateTime, TimeZone, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Bybit API base URL
const BYBIT_API_URL: &str = "https://api.bybit.com";

/// Errors that can occur when interacting with Bybit API
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("Failed to parse response: {0}")]
    ParseError(#[from] serde_json::Error),

    #[error("API error: {message} (code: {code})")]
    ApiError { code: i32, message: String },

    #[error("Invalid interval: {0}")]
    InvalidInterval(String),
}

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
pub struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
}

/// Kline response data
#[derive(Debug, Deserialize)]
pub struct KlineResponse {
    pub symbol: String,
    pub category: String,
    pub list: Vec<KlineData>,
}

/// Individual kline/candle data
/// Format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct KlineData(
    pub String, // start time (ms)
    pub String, // open
    pub String, // high
    pub String, // low
    pub String, // close
    pub String, // volume
    pub String, // turnover
);

impl KlineData {
    /// Convert to Candle struct
    pub fn to_candle(&self) -> Result<Candle, BybitError> {
        let timestamp_ms: i64 = self.0.parse().map_err(|_| BybitError::ApiError {
            code: -1,
            message: "Failed to parse timestamp".to_string(),
        })?;

        let timestamp = Utc.timestamp_millis_opt(timestamp_ms).unwrap();

        Ok(Candle {
            timestamp,
            open: self.1.parse().unwrap_or(0.0),
            high: self.2.parse().unwrap_or(0.0),
            low: self.3.parse().unwrap_or(0.0),
            close: self.4.parse().unwrap_or(0.0),
            volume: self.5.parse().unwrap_or(0.0),
        })
    }
}

/// Bybit API client for fetching market data
#[derive(Debug, Clone)]
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: BYBIT_API_URL.to_string(),
        }
    }

    /// Create a new Bybit client with custom base URL (for testing)
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Get klines (candlestick data) from Bybit
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, M, W)
    /// * `limit` - Number of klines to fetch (max 1000)
    ///
    /// # Returns
    ///
    /// Vector of Candle structs ordered from oldest to newest
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Candle>, BybitError> {
        // Validate interval
        let valid_intervals = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "M", "W"];
        if !valid_intervals.contains(&interval) {
            return Err(BybitError::InvalidInterval(interval.to_string()));
        }

        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit.min(1000)
        );

        let response = self.client.get(&url).send().await?;
        let api_response: BybitResponse<KlineResponse> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        // Convert to Candles and reverse (API returns newest first)
        let mut candles: Vec<Candle> = api_response
            .result
            .list
            .iter()
            .filter_map(|k| k.to_candle().ok())
            .collect();

        candles.reverse();

        Ok(candles)
    }

    /// Get klines with specific start and end times
    pub async fn get_klines_range(
        &self,
        symbol: &str,
        interval: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<Candle>, BybitError> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&start={}&end={}&limit=1000",
            self.base_url,
            symbol,
            interval,
            start.timestamp_millis(),
            end.timestamp_millis()
        );

        let response = self.client.get(&url).send().await?;
        let api_response: BybitResponse<KlineResponse> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        let mut candles: Vec<Candle> = api_response
            .result
            .list
            .iter()
            .filter_map(|k| k.to_candle().ok())
            .collect();

        candles.reverse();

        Ok(candles)
    }

    /// Get ticker information for a symbol
    pub async fn get_ticker(&self, symbol: &str) -> Result<TickerInfo, BybitError> {
        let url = format!(
            "{}/v5/market/tickers?category=linear&symbol={}",
            self.base_url, symbol
        );

        let response = self.client.get(&url).send().await?;
        let api_response: BybitResponse<TickerResponse> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        api_response
            .result
            .list
            .into_iter()
            .next()
            .ok_or_else(|| BybitError::ApiError {
                code: -1,
                message: "Ticker not found".to_string(),
            })
    }

    /// Get available symbols
    pub async fn get_symbols(&self) -> Result<Vec<String>, BybitError> {
        let url = format!("{}/v5/market/tickers?category=linear", self.base_url);

        let response = self.client.get(&url).send().await?;
        let api_response: BybitResponse<TickerResponse> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        Ok(api_response
            .result
            .list
            .into_iter()
            .map(|t| t.symbol)
            .collect())
    }
}

/// Ticker response wrapper
#[derive(Debug, Deserialize)]
pub struct TickerResponse {
    pub category: String,
    pub list: Vec<TickerInfo>,
}

/// Ticker information
#[derive(Debug, Deserialize, Clone)]
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

impl TickerInfo {
    /// Get last price as f64
    pub fn last_price_f64(&self) -> f64 {
        self.last_price.parse().unwrap_or(0.0)
    }

    /// Get 24h price change percentage
    pub fn price_change_pct(&self) -> f64 {
        self.price_24h_pcnt.parse::<f64>().unwrap_or(0.0) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_data_parse() {
        let kline = KlineData(
            "1704067200000".to_string(),
            "42000.0".to_string(),
            "42500.0".to_string(),
            "41500.0".to_string(),
            "42100.0".to_string(),
            "1000.0".to_string(),
            "42000000.0".to_string(),
        );

        let candle = kline.to_candle().unwrap();
        assert_eq!(candle.open, 42000.0);
        assert_eq!(candle.high, 42500.0);
        assert_eq!(candle.low, 41500.0);
        assert_eq!(candle.close, 42100.0);
        assert_eq!(candle.volume, 1000.0);
    }
}
