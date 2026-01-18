//! Bybit API client for cryptocurrency data
//!
//! Fetches OHLCV data from Bybit exchange.

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Bybit API error
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),
    #[error("API error: {0}")]
    ApiError(String),
    #[error("Parse error: {0}")]
    ParseError(String),
}

/// OHLCV candle data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Timestamp in milliseconds
    pub timestamp: i64,
    /// Open price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Close price
    pub close: f64,
    /// Volume
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

    /// Calculate return from open to close
    pub fn return_oc(&self) -> f64 {
        (self.close - self.open) / self.open
    }

    /// Calculate log return
    pub fn log_return(&self) -> f64 {
        (self.close / self.open).ln()
    }

    /// Calculate true range
    pub fn true_range(&self, prev_close: Option<f64>) -> f64 {
        let hl = self.high - self.low;
        match prev_close {
            Some(pc) => {
                let hc = (self.high - pc).abs();
                let lc = (self.low - pc).abs();
                hl.max(hc).max(lc)
            }
            None => hl,
        }
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

#[derive(Debug, Deserialize)]
struct BybitResult {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

/// Bybit API client
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Create a new Bybit client with custom base URL
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Fetch kline/candlestick data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Kline interval (e.g., "60" for 1 hour)
    /// * `limit` - Number of candles to fetch (max 1000)
    ///
    /// # Returns
    /// Vector of Candle structs, sorted by timestamp ascending
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>, BybitError> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit.min(1000)
        );

        let response: BybitResponse = self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError(response.ret_msg));
        }

        let mut candles: Vec<Candle> = response
            .result
            .list
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 7 {
                    Some(Candle {
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

        // Sort by timestamp ascending (Bybit returns descending)
        candles.sort_by_key(|c| c.timestamp);

        Ok(candles)
    }

    /// Fetch data for multiple symbols
    pub async fn fetch_multi_asset(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: usize,
    ) -> Result<HashMap<String, Vec<Candle>>, BybitError> {
        let mut result = HashMap::new();

        for symbol in symbols {
            let candles = self.fetch_klines(symbol, interval, limit).await?;
            result.insert(symbol.to_string(), candles);
        }

        Ok(result)
    }

    /// Get available trading pairs
    pub async fn get_symbols(&self) -> Result<Vec<String>, BybitError> {
        let url = format!("{}/v5/market/instruments-info?category=spot", self.base_url);

        #[derive(Deserialize)]
        struct InstrumentsResponse {
            #[serde(rename = "retCode")]
            ret_code: i32,
            #[serde(rename = "retMsg")]
            ret_msg: String,
            result: InstrumentsResult,
        }

        #[derive(Deserialize)]
        struct InstrumentsResult {
            list: Vec<Instrument>,
        }

        #[derive(Deserialize)]
        struct Instrument {
            symbol: String,
        }

        let response: InstrumentsResponse = self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError(response.ret_msg));
        }

        Ok(response.result.list.into_iter().map(|i| i.symbol).collect())
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_return() {
        let candle = Candle::new(0, 100.0, 110.0, 95.0, 105.0, 1000.0, 100000.0);

        assert!((candle.return_oc() - 0.05).abs() < 1e-10);
        assert!((candle.log_return() - 0.04879016417).abs() < 1e-6);
    }

    #[test]
    fn test_candle_true_range() {
        let candle = Candle::new(0, 100.0, 110.0, 95.0, 105.0, 1000.0, 100000.0);

        // Without previous close
        assert!((candle.true_range(None) - 15.0).abs() < 1e-10);

        // With previous close
        assert!((candle.true_range(Some(90.0)) - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_client_creation() {
        let client = BybitClient::new();
        assert_eq!(client.base_url, "https://api.bybit.com");

        let custom = BybitClient::with_base_url("https://custom.api");
        assert_eq!(custom.base_url, "https://custom.api");
    }
}
