//! Bybit API client for cryptocurrency data.

use chrono::{DateTime, Utc};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur when interacting with the Bybit API.
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Parse error: {0}")]
    ParseError(String),
}

/// A single candlestick/kline data point.
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

/// Response from Bybit API.
#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: T,
}

#[derive(Debug, Deserialize)]
struct KlineResult {
    list: Vec<Vec<String>>,
}

/// Bybit API client for fetching market data.
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client.
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Create a new Bybit client for testnet.
    pub fn testnet() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api-testnet.bybit.com".to_string(),
        }
    }

    /// Fetch kline/candlestick data.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
    /// * `limit` - Number of klines to fetch (max 1000)
    ///
    /// # Returns
    ///
    /// Vector of Kline data points.
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Kline>, BybitError> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.min(1000).to_string()),
            ])
            .send()?;

        let data: BybitResponse<KlineResult> = response.json()?;

        if data.ret_code != 0 {
            return Err(BybitError::ApiError(data.ret_msg));
        }

        let mut klines = Vec::new();
        for item in data.result.list {
            if item.len() >= 7 {
                let timestamp_ms: i64 = item[0]
                    .parse()
                    .map_err(|e| BybitError::ParseError(format!("Invalid timestamp: {}", e)))?;

                let timestamp = DateTime::from_timestamp_millis(timestamp_ms)
                    .ok_or_else(|| BybitError::ParseError("Invalid timestamp".to_string()))?;

                let kline = Kline {
                    timestamp,
                    open: item[1].parse().unwrap_or(0.0),
                    high: item[2].parse().unwrap_or(0.0),
                    low: item[3].parse().unwrap_or(0.0),
                    close: item[4].parse().unwrap_or(0.0),
                    volume: item[5].parse().unwrap_or(0.0),
                    turnover: item[6].parse().unwrap_or(0.0),
                };
                klines.push(kline);
            }
        }

        // Sort by timestamp (API returns newest first)
        klines.sort_by_key(|k| k.timestamp);

        Ok(klines)
    }

    /// Fetch current ticker data.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    pub fn fetch_ticker(&self, symbol: &str) -> Result<Ticker, BybitError> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[("category", "linear"), ("symbol", symbol)])
            .send()?;

        let data: BybitResponse<TickerResult> = response.json()?;

        if data.ret_code != 0 {
            return Err(BybitError::ApiError(data.ret_msg));
        }

        data.result
            .list
            .into_iter()
            .next()
            .ok_or_else(|| BybitError::ApiError("No ticker data".to_string()))
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Ticker data.
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
}

#[derive(Debug, Deserialize)]
struct TickerResult {
    list: Vec<Ticker>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = BybitClient::new();
        assert_eq!(client.base_url, "https://api.bybit.com");
    }

    #[test]
    fn test_testnet_client() {
        let client = BybitClient::testnet();
        assert_eq!(client.base_url, "https://api-testnet.bybit.com");
    }
}
