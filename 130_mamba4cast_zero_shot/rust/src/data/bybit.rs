//! Bybit API client for cryptocurrency data.
//!
//! This module provides functionality to fetch market data from Bybit exchange.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur when fetching Bybit data.
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Parse error: {0}")]
    ParseError(String),
}

/// Kline (candlestick) data from Bybit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KlineData {
    pub timestamp: DateTime<Utc>,
    pub open: f32,
    pub high: f32,
    pub low: f32,
    pub close: f32,
    pub volume: f32,
    pub turnover: f32,
}

/// Orderbook data from Bybit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderbookData {
    pub symbol: String,
    pub bids: Vec<(f32, f32)>, // (price, quantity)
    pub asks: Vec<(f32, f32)>,
    pub timestamp: i64,
}

/// API response structure.
#[derive(Debug, Deserialize)]
struct ApiResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: T,
}

/// Kline API result.
#[derive(Debug, Deserialize)]
struct KlineResult {
    list: Vec<Vec<String>>,
}

/// Orderbook API result.
#[derive(Debug, Deserialize)]
struct OrderbookResult {
    s: String,
    b: Vec<Vec<String>>,
    a: Vec<Vec<String>>,
    ts: i64,
}

/// Client for interacting with Bybit API.
pub struct BybitClient {
    base_url: String,
    client: reqwest::blocking::Client,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Create a new Bybit client.
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
        }
    }

    /// Fetch kline data for a symbol.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Kline interval in minutes ("1", "5", "15", "60", etc.)
    /// * `limit` - Number of klines to fetch (max 1000)
    ///
    /// # Returns
    ///
    /// Vector of KlineData sorted by timestamp (oldest first)
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<KlineData>, BybitError> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let response = self.client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.min(1000).to_string()),
            ])
            .send()?;

        let api_response: ApiResponse<KlineResult> = response.json()?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError(api_response.ret_msg));
        }

        let mut klines: Vec<KlineData> = api_response
            .result
            .list
            .into_iter()
            .filter_map(|row| self.parse_kline_row(&row).ok())
            .collect();

        // Sort by timestamp (oldest first)
        klines.sort_by_key(|k| k.timestamp);

        Ok(klines)
    }

    /// Parse a single kline row from API response.
    fn parse_kline_row(&self, row: &[String]) -> Result<KlineData, BybitError> {
        if row.len() < 7 {
            return Err(BybitError::ParseError("Invalid row length".to_string()));
        }

        let timestamp_ms: i64 = row[0].parse()
            .map_err(|_| BybitError::ParseError("Invalid timestamp".to_string()))?;

        let timestamp = DateTime::from_timestamp_millis(timestamp_ms)
            .unwrap_or_else(Utc::now);

        Ok(KlineData {
            timestamp,
            open: row[1].parse().unwrap_or(0.0),
            high: row[2].parse().unwrap_or(0.0),
            low: row[3].parse().unwrap_or(0.0),
            close: row[4].parse().unwrap_or(0.0),
            volume: row[5].parse().unwrap_or(0.0),
            turnover: row[6].parse().unwrap_or(0.0),
        })
    }

    /// Fetch current orderbook for a symbol.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair
    /// * `limit` - Orderbook depth
    pub fn fetch_orderbook(
        &self,
        symbol: &str,
        limit: usize,
    ) -> Result<OrderbookData, BybitError> {
        let url = format!("{}/v5/market/orderbook", self.base_url);

        let response = self.client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("limit", &limit.to_string()),
            ])
            .send()?;

        let api_response: ApiResponse<OrderbookResult> = response.json()?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError(api_response.ret_msg));
        }

        let result = api_response.result;

        let bids: Vec<(f32, f32)> = result.b
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 2 {
                    Some((
                        row[0].parse().unwrap_or(0.0),
                        row[1].parse().unwrap_or(0.0),
                    ))
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<(f32, f32)> = result.a
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 2 {
                    Some((
                        row[0].parse().unwrap_or(0.0),
                        row[1].parse().unwrap_or(0.0),
                    ))
                } else {
                    None
                }
            })
            .collect();

        Ok(OrderbookData {
            symbol: result.s,
            bids,
            asks,
            timestamp: result.ts,
        })
    }

    /// Get close prices from kline data.
    pub fn extract_close_prices(klines: &[KlineData]) -> Vec<f32> {
        klines.iter().map(|k| k.close).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = BybitClient::new();
        assert_eq!(client.base_url, "https://api.bybit.com");
    }

    // Note: Integration tests require network access
    // #[test]
    // fn test_fetch_klines() {
    //     let client = BybitClient::new();
    //     let klines = client.fetch_klines("BTCUSDT", "60", 100).unwrap();
    //     assert!(!klines.is_empty());
    // }
}
