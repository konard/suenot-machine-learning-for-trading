//! Bybit Exchange API Client
//!
//! This module provides a client for fetching market data from the Bybit
//! cryptocurrency exchange.

use super::{Candle, Interval, OrderBook, OrderBookLevel, Trade, TradeSide};
use chrono::{DateTime, TimeZone, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Bybit API base URL
const BYBIT_API_BASE: &str = "https://api.bybit.com";

/// Bybit API errors
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("API error: {code} - {message}")]
    ApiError { code: i32, message: String },

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// Result type for Bybit operations
pub type Result<T> = std::result::Result<T, BybitError>;

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApiResponse<T> {
    ret_code: i32,
    ret_msg: String,
    result: T,
}

/// Kline response data
#[derive(Debug, Deserialize)]
struct KlineResult {
    category: String,
    symbol: String,
    list: Vec<Vec<String>>,
}

/// Order book response
#[derive(Debug, Deserialize)]
struct OrderBookResult {
    s: String,   // symbol
    b: Vec<Vec<String>>, // bids
    a: Vec<Vec<String>>, // asks
    ts: u64,     // timestamp
}

/// Bybit API Client
///
/// # Example
///
/// ```rust,no_run
/// use hopfield_trading::data::bybit::BybitClient;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let client = BybitClient::new();
///
///     // Get 100 hourly candles for BTC/USDT
///     let candles = client.get_klines("BTCUSDT", "60", 100).await?;
///
///     for candle in candles.iter().take(5) {
///         println!("Close: {:.2}", candle.close);
///     }
///
///     Ok(())
/// }
/// ```
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
            base_url: BYBIT_API_BASE.to_string(),
        }
    }

    /// Create a client with custom base URL (for testing)
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Get kline/candlestick data
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Time interval (e.g., "1", "60", "D")
    /// * `limit` - Number of candles to fetch (max 200)
    ///
    /// # Intervals
    ///
    /// - Minutes: "1", "3", "5", "15", "30"
    /// - Hours: "60", "120", "240", "360", "720"
    /// - Days/Weeks/Months: "D", "W", "M"
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>> {
        let limit = limit.min(200);

        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response: ApiResponse<KlineResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let candles = response
            .result
            .list
            .into_iter()
            .filter_map(|row| self.parse_kline_row(&row).ok())
            .collect();

        Ok(candles)
    }

    /// Get klines using Interval enum
    pub async fn get_klines_interval(
        &self,
        symbol: &str,
        interval: Interval,
        limit: usize,
    ) -> Result<Vec<Candle>> {
        self.get_klines(symbol, interval.to_bybit_string(), limit).await
    }

    /// Parse a kline row from the API response
    fn parse_kline_row(&self, row: &[String]) -> Result<Candle> {
        if row.len() < 7 {
            return Err(BybitError::ParseError(
                "Invalid kline row length".to_string(),
            ));
        }

        let timestamp_ms: i64 = row[0]
            .parse()
            .map_err(|_| BybitError::ParseError("Invalid timestamp".to_string()))?;

        let timestamp = Utc
            .timestamp_millis_opt(timestamp_ms)
            .single()
            .ok_or_else(|| BybitError::ParseError("Invalid timestamp value".to_string()))?;

        let open: f64 = row[1]
            .parse()
            .map_err(|_| BybitError::ParseError("Invalid open price".to_string()))?;

        let high: f64 = row[2]
            .parse()
            .map_err(|_| BybitError::ParseError("Invalid high price".to_string()))?;

        let low: f64 = row[3]
            .parse()
            .map_err(|_| BybitError::ParseError("Invalid low price".to_string()))?;

        let close: f64 = row[4]
            .parse()
            .map_err(|_| BybitError::ParseError("Invalid close price".to_string()))?;

        let volume: f64 = row[5]
            .parse()
            .map_err(|_| BybitError::ParseError("Invalid volume".to_string()))?;

        let turnover: f64 = row[6]
            .parse()
            .map_err(|_| BybitError::ParseError("Invalid turnover".to_string()))?;

        Ok(Candle::new(timestamp, open, high, low, close, volume, turnover))
    }

    /// Get order book depth
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair
    /// * `limit` - Depth limit (1, 25, 50, 100, 200)
    pub async fn get_orderbook(&self, symbol: &str, limit: usize) -> Result<OrderBook> {
        let limit = match limit {
            l if l <= 1 => 1,
            l if l <= 25 => 25,
            l if l <= 50 => 50,
            l if l <= 100 => 100,
            _ => 200,
        };

        let url = format!(
            "{}/v5/market/orderbook?category=linear&symbol={}&limit={}",
            self.base_url, symbol, limit
        );

        let response: ApiResponse<OrderBookResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let bids = response
            .result
            .b
            .into_iter()
            .filter_map(|row| self.parse_orderbook_level(&row).ok())
            .collect();

        let asks = response
            .result
            .a
            .into_iter()
            .filter_map(|row| self.parse_orderbook_level(&row).ok())
            .collect();

        let timestamp = Utc
            .timestamp_millis_opt(response.result.ts as i64)
            .single()
            .unwrap_or_else(Utc::now);

        Ok(OrderBook {
            symbol: response.result.s,
            bids,
            asks,
            timestamp,
        })
    }

    /// Parse an order book level
    fn parse_orderbook_level(&self, row: &[String]) -> Result<OrderBookLevel> {
        if row.len() < 2 {
            return Err(BybitError::ParseError(
                "Invalid orderbook level".to_string(),
            ));
        }

        let price: f64 = row[0]
            .parse()
            .map_err(|_| BybitError::ParseError("Invalid price".to_string()))?;

        let qty: f64 = row[1]
            .parse()
            .map_err(|_| BybitError::ParseError("Invalid quantity".to_string()))?;

        Ok(OrderBookLevel { price, qty })
    }

    /// Get ticker information
    pub async fn get_ticker(&self, symbol: &str) -> Result<TickerInfo> {
        let url = format!(
            "{}/v5/market/tickers?category=linear&symbol={}",
            self.base_url, symbol
        );

        let response: ApiResponse<TickerResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        response
            .result
            .list
            .into_iter()
            .next()
            .ok_or_else(|| BybitError::ParseError("No ticker data".to_string()))
    }

    /// Get multiple symbols' tickers
    pub async fn get_tickers(&self) -> Result<Vec<TickerInfo>> {
        let url = format!(
            "{}/v5/market/tickers?category=linear",
            self.base_url
        );

        let response: ApiResponse<TickerResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        Ok(response.result.list)
    }

    /// Get server time
    pub async fn get_server_time(&self) -> Result<DateTime<Utc>> {
        let url = format!("{}/v5/market/time", self.base_url);

        let response: ApiResponse<ServerTimeResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let timestamp_ms: i64 = response
            .result
            .time_now
            .parse()
            .map_err(|_| BybitError::ParseError("Invalid server time".to_string()))?;

        Utc.timestamp_millis_opt(timestamp_ms)
            .single()
            .ok_or_else(|| BybitError::ParseError("Invalid timestamp".to_string()))
    }
}

/// Ticker information
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TickerInfo {
    pub symbol: String,
    #[serde(deserialize_with = "deserialize_string_to_f64")]
    pub last_price: f64,
    #[serde(deserialize_with = "deserialize_string_to_f64")]
    pub index_price: f64,
    #[serde(deserialize_with = "deserialize_string_to_f64")]
    pub mark_price: f64,
    #[serde(deserialize_with = "deserialize_string_to_f64")]
    pub prev_price24h: f64,
    #[serde(deserialize_with = "deserialize_string_to_f64")]
    pub price24h_pcnt: f64,
    #[serde(deserialize_with = "deserialize_string_to_f64")]
    pub high_price24h: f64,
    #[serde(deserialize_with = "deserialize_string_to_f64")]
    pub low_price24h: f64,
    #[serde(deserialize_with = "deserialize_string_to_f64")]
    pub volume24h: f64,
    #[serde(deserialize_with = "deserialize_string_to_f64")]
    pub turnover24h: f64,
}

#[derive(Debug, Deserialize)]
struct TickerResult {
    list: Vec<TickerInfo>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ServerTimeResult {
    time_now: String,
}

/// Helper function to deserialize string to f64
fn deserialize_string_to_f64<'de, D>(deserializer: D) -> std::result::Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    s.parse().map_err(serde::de::Error::custom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = BybitClient::new();
        assert_eq!(client.base_url, BYBIT_API_BASE);
    }

    // Integration tests would require network access
    // Run with: cargo test -- --ignored

    #[tokio::test]
    #[ignore]
    async fn test_get_klines() {
        let client = BybitClient::new();
        let candles = client.get_klines("BTCUSDT", "60", 10).await.unwrap();
        assert!(!candles.is_empty());
        assert!(candles.len() <= 10);
    }

    #[tokio::test]
    #[ignore]
    async fn test_get_orderbook() {
        let client = BybitClient::new();
        let orderbook = client.get_orderbook("BTCUSDT", 25).await.unwrap();
        assert_eq!(orderbook.symbol, "BTCUSDT");
        assert!(!orderbook.bids.is_empty());
        assert!(!orderbook.asks.is_empty());
    }

    #[tokio::test]
    #[ignore]
    async fn test_get_ticker() {
        let client = BybitClient::new();
        let ticker = client.get_ticker("BTCUSDT").await.unwrap();
        assert_eq!(ticker.symbol, "BTCUSDT");
        assert!(ticker.last_price > 0.0);
    }
}
