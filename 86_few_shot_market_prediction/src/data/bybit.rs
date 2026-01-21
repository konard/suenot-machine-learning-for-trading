//! Bybit API client for fetching market data

use super::types::{FundingRate, Kline, OrderBook, OrderBookLevel, Ticker, Trade};
use chrono::{DateTime, TimeZone, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

/// Bybit API errors
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("API error: {code} - {message}")]
    ApiError { code: i32, message: String },

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Rate limited")]
    RateLimited,
}

/// Bybit API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitConfig {
    /// Base URL for the API
    pub base_url: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Whether to use testnet
    pub testnet: bool,
}

impl Default for BybitConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            timeout_secs: 30,
            testnet: false,
        }
    }
}

impl BybitConfig {
    /// Create config for testnet
    pub fn testnet() -> Self {
        Self {
            base_url: "https://api-testnet.bybit.com".to_string(),
            testnet: true,
            ..Default::default()
        }
    }
}

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
struct ApiResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: Option<T>,
}

/// Kline response data
#[derive(Debug, Deserialize)]
struct KlineResult {
    list: Vec<Vec<String>>,
}

/// Ticker response data
#[derive(Debug, Deserialize)]
struct TickerResult {
    list: Vec<TickerData>,
}

#[derive(Debug, Deserialize)]
struct TickerData {
    symbol: String,
    #[serde(rename = "lastPrice")]
    last_price: String,
    #[serde(rename = "highPrice24h")]
    high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    low_price_24h: String,
    #[serde(rename = "volume24h")]
    volume_24h: String,
    #[serde(rename = "turnover24h")]
    turnover_24h: String,
    #[serde(rename = "price24hPcnt")]
    price_24h_pcnt: String,
}

/// Order book response data
#[derive(Debug, Deserialize)]
struct OrderBookResult {
    b: Vec<Vec<String>>,  // bids
    a: Vec<Vec<String>>,  // asks
    ts: u64,
}

/// Bybit API client
pub struct BybitClient {
    config: BybitConfig,
    client: Client,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new(config: BybitConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self { config, client }
    }

    /// Create client with default config
    pub fn default_client() -> Self {
        Self::new(BybitConfig::default())
    }

    /// Fetch klines (candlestick data)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: Option<u32>,
    ) -> Result<Vec<Kline>, BybitError> {
        let limit = limit.unwrap_or(200);
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.config.base_url, symbol, interval, limit
        );

        let response: ApiResponse<KlineResult> = self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let result = response.result.ok_or_else(|| {
            BybitError::ParseError("Missing result in response".to_string())
        })?;

        let mut klines = Vec::new();
        for item in result.list {
            if item.len() >= 7 {
                let timestamp_ms: i64 = item[0].parse().unwrap_or(0);
                let timestamp = Utc.timestamp_millis_opt(timestamp_ms).unwrap();

                klines.push(Kline {
                    timestamp,
                    open: item[1].parse().unwrap_or(0.0),
                    high: item[2].parse().unwrap_or(0.0),
                    low: item[3].parse().unwrap_or(0.0),
                    close: item[4].parse().unwrap_or(0.0),
                    volume: item[5].parse().unwrap_or(0.0),
                    turnover: item[6].parse().unwrap_or(0.0),
                });
            }
        }

        // Bybit returns newest first, reverse for chronological order
        klines.reverse();
        Ok(klines)
    }

    /// Fetch ticker data
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker, BybitError> {
        let url = format!(
            "{}/v5/market/tickers?category=linear&symbol={}",
            self.config.base_url, symbol
        );

        let response: ApiResponse<TickerResult> = self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let result = response.result.ok_or_else(|| {
            BybitError::ParseError("Missing result in response".to_string())
        })?;

        let data = result.list.into_iter().next().ok_or_else(|| {
            BybitError::ParseError("No ticker data returned".to_string())
        })?;

        Ok(Ticker {
            symbol: data.symbol,
            last_price: data.last_price.parse().unwrap_or(0.0),
            high_24h: data.high_price_24h.parse().unwrap_or(0.0),
            low_24h: data.low_price_24h.parse().unwrap_or(0.0),
            volume_24h: data.volume_24h.parse().unwrap_or(0.0),
            turnover_24h: data.turnover_24h.parse().unwrap_or(0.0),
            price_change_pct: data.price_24h_pcnt.parse().unwrap_or(0.0),
        })
    }

    /// Fetch order book
    pub async fn get_orderbook(&self, symbol: &str, limit: Option<u32>) -> Result<OrderBook, BybitError> {
        let limit = limit.unwrap_or(50);
        let url = format!(
            "{}/v5/market/orderbook?category=linear&symbol={}&limit={}",
            self.config.base_url, symbol, limit
        );

        let response: ApiResponse<OrderBookResult> = self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let result = response.result.ok_or_else(|| {
            BybitError::ParseError("Missing result in response".to_string())
        })?;

        let bids: Vec<OrderBookLevel> = result
            .b
            .iter()
            .filter_map(|level| {
                if level.len() >= 2 {
                    Some(OrderBookLevel {
                        price: level[0].parse().unwrap_or(0.0),
                        quantity: level[1].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<OrderBookLevel> = result
            .a
            .iter()
            .filter_map(|level| {
                if level.len() >= 2 {
                    Some(OrderBookLevel {
                        price: level[0].parse().unwrap_or(0.0),
                        quantity: level[1].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        let timestamp = Utc.timestamp_millis_opt(result.ts as i64).unwrap();

        Ok(OrderBook {
            timestamp,
            bids,
            asks,
        })
    }

    /// Get base URL
    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    /// Check if using testnet
    pub fn is_testnet(&self) -> bool {
        self.config.testnet
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = BybitConfig::default();
        assert!(!config.testnet);
        assert!(config.base_url.contains("api.bybit.com"));
    }

    #[test]
    fn test_config_testnet() {
        let config = BybitConfig::testnet();
        assert!(config.testnet);
        assert!(config.base_url.contains("testnet"));
    }
}
