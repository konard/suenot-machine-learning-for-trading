//! Bybit API Client
//!
//! This module provides a client for interacting with the Bybit cryptocurrency exchange API.
//! It supports fetching market data, recent trades, and order book information.
//!
//! # Example
//!
//! ```rust,no_run
//! use crypto_embeddings::BybitClient;
//!
//! #[tokio::main]
//! async fn main() {
//!     let client = BybitClient::new();
//!     let trades = client.get_recent_trades("BTCUSDT", 100).await.unwrap();
//!     println!("Got {} trades", trades.len());
//! }
//! ```

use serde::{Deserialize, Serialize};
use crate::utils::{Result, CryptoEmbeddingsError};

/// Bybit API base URL
const BYBIT_API_BASE: &str = "https://api.bybit.com";

/// A trade record from Bybit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade execution ID
    pub exec_id: String,
    /// Trading symbol (e.g., "BTCUSDT")
    pub symbol: String,
    /// Trade price
    pub price: f64,
    /// Trade quantity
    pub qty: f64,
    /// Trade side: "Buy" or "Sell"
    pub side: String,
    /// Trade timestamp (milliseconds)
    pub time: i64,
    /// Whether taker is buyer
    pub is_buyer_maker: bool,
}

/// Order book entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookEntry {
    /// Price level
    pub price: f64,
    /// Quantity at this level
    pub qty: f64,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Trading symbol
    pub symbol: String,
    /// Bid orders (buy side)
    pub bids: Vec<OrderBookEntry>,
    /// Ask orders (sell side)
    pub asks: Vec<OrderBookEntry>,
    /// Timestamp
    pub timestamp: i64,
}

/// Kline/Candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Start time (milliseconds)
    pub start_time: i64,
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
    /// Turnover
    pub turnover: f64,
}

/// Ticker information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Trading symbol
    pub symbol: String,
    /// Last traded price
    pub last_price: f64,
    /// 24h high price
    pub high_price_24h: f64,
    /// 24h low price
    pub low_price_24h: f64,
    /// 24h price change percentage
    pub price_change_24h: f64,
    /// 24h volume
    pub volume_24h: f64,
    /// 24h turnover
    pub turnover_24h: f64,
}

/// Bybit API client
pub struct BybitClient {
    client: reqwest::Client,
    base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: BYBIT_API_BASE.to_string(),
        }
    }

    /// Create a client with custom base URL (useful for testnet)
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Get recent trades for a symbol
    pub async fn get_recent_trades(&self, symbol: &str, limit: usize) -> Result<Vec<Trade>> {
        let url = format!("{}/v5/market/recent-trade", self.base_url);

        let resp = self.client
            .get(&url)
            .query(&[
                ("category", "spot"),
                ("symbol", symbol),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?;

        let json: serde_json::Value = resp.json().await?;

        if json["retCode"].as_i64() != Some(0) {
            return Err(CryptoEmbeddingsError::ApiError(
                json["retMsg"].as_str().unwrap_or("Unknown error").to_string()
            ));
        }

        let trades_data = json["result"]["list"]
            .as_array()
            .ok_or_else(|| CryptoEmbeddingsError::ApiError("Invalid response format".to_string()))?;

        let mut trades = Vec::with_capacity(trades_data.len());
        for item in trades_data {
            let trade = Trade {
                exec_id: item["execId"].as_str().unwrap_or("").to_string(),
                symbol: item["symbol"].as_str().unwrap_or("").to_string(),
                price: item["price"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                qty: item["size"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                side: item["side"].as_str().unwrap_or("").to_string(),
                time: item["time"].as_str().unwrap_or("0").parse().unwrap_or(0),
                is_buyer_maker: item["side"].as_str() == Some("Sell"),
            };
            trades.push(trade);
        }

        Ok(trades)
    }

    /// Get order book for a symbol
    pub async fn get_orderbook(&self, symbol: &str, limit: usize) -> Result<OrderBook> {
        let url = format!("{}/v5/market/orderbook", self.base_url);

        let resp = self.client
            .get(&url)
            .query(&[
                ("category", "spot"),
                ("symbol", symbol),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?;

        let json: serde_json::Value = resp.json().await?;

        if json["retCode"].as_i64() != Some(0) {
            return Err(CryptoEmbeddingsError::ApiError(
                json["retMsg"].as_str().unwrap_or("Unknown error").to_string()
            ));
        }

        let result = &json["result"];

        let parse_levels = |levels: &serde_json::Value| -> Vec<OrderBookEntry> {
            levels.as_array()
                .map(|arr| {
                    arr.iter().map(|item| {
                        OrderBookEntry {
                            price: item[0].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                            qty: item[1].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                        }
                    }).collect()
                })
                .unwrap_or_default()
        };

        Ok(OrderBook {
            symbol: symbol.to_string(),
            bids: parse_levels(&result["b"]),
            asks: parse_levels(&result["a"]),
            timestamp: result["ts"].as_str().unwrap_or("0").parse().unwrap_or(0),
        })
    }

    /// Get kline/candlestick data
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let resp = self.client
            .get(&url)
            .query(&[
                ("category", "spot"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?;

        let json: serde_json::Value = resp.json().await?;

        if json["retCode"].as_i64() != Some(0) {
            return Err(CryptoEmbeddingsError::ApiError(
                json["retMsg"].as_str().unwrap_or("Unknown error").to_string()
            ));
        }

        let klines_data = json["result"]["list"]
            .as_array()
            .ok_or_else(|| CryptoEmbeddingsError::ApiError("Invalid response format".to_string()))?;

        let mut klines = Vec::with_capacity(klines_data.len());
        for item in klines_data {
            let kline = Kline {
                start_time: item[0].as_str().unwrap_or("0").parse().unwrap_or(0),
                open: item[1].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                high: item[2].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                low: item[3].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                close: item[4].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                volume: item[5].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                turnover: item[6].as_str().unwrap_or("0").parse().unwrap_or(0.0),
            };
            klines.push(kline);
        }

        Ok(klines)
    }

    /// Get ticker information for a symbol
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        let resp = self.client
            .get(&url)
            .query(&[
                ("category", "spot"),
                ("symbol", symbol),
            ])
            .send()
            .await?;

        let json: serde_json::Value = resp.json().await?;

        if json["retCode"].as_i64() != Some(0) {
            return Err(CryptoEmbeddingsError::ApiError(
                json["retMsg"].as_str().unwrap_or("Unknown error").to_string()
            ));
        }

        let item = &json["result"]["list"][0];

        Ok(Ticker {
            symbol: item["symbol"].as_str().unwrap_or("").to_string(),
            last_price: item["lastPrice"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
            high_price_24h: item["highPrice24h"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
            low_price_24h: item["lowPrice24h"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
            price_change_24h: item["price24hPcnt"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
            volume_24h: item["volume24h"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
            turnover_24h: item["turnover24h"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
        })
    }

    /// Get list of all trading symbols
    pub async fn get_symbols(&self) -> Result<Vec<String>> {
        let url = format!("{}/v5/market/instruments-info", self.base_url);

        let resp = self.client
            .get(&url)
            .query(&[("category", "spot")])
            .send()
            .await?;

        let json: serde_json::Value = resp.json().await?;

        if json["retCode"].as_i64() != Some(0) {
            return Err(CryptoEmbeddingsError::ApiError(
                json["retMsg"].as_str().unwrap_or("Unknown error").to_string()
            ));
        }

        let symbols = json["result"]["list"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| item["symbol"].as_str())
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default();

        Ok(symbols)
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

    #[tokio::test]
    async fn test_client_creation() {
        let client = BybitClient::new();
        assert_eq!(client.base_url, BYBIT_API_BASE);
    }

    #[tokio::test]
    async fn test_custom_base_url() {
        let client = BybitClient::with_base_url("https://api-testnet.bybit.com");
        assert_eq!(client.base_url, "https://api-testnet.bybit.com");
    }
}
