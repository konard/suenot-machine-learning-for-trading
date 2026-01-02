//! Bybit REST API client
//!
//! Implements the Bybit V5 API for fetching market data.

use crate::data::{Candle, OrderBook, OrderBookLevel, Trade};
use chrono::{DateTime, Utc};
use hmac::{Hmac, Mac};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

/// Bybit API configuration
#[derive(Debug, Clone)]
pub struct BybitConfig {
    /// Base URL for the API
    pub base_url: String,
    /// API key (optional for public endpoints)
    pub api_key: Option<String>,
    /// API secret (optional for public endpoints)
    pub api_secret: Option<String>,
    /// Request timeout in seconds
    pub timeout_secs: u64,
}

impl Default for BybitConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            api_key: None,
            api_secret: None,
            timeout_secs: 30,
        }
    }
}

impl BybitConfig {
    /// Create config for testnet
    pub fn testnet() -> Self {
        Self {
            base_url: "https://api-testnet.bybit.com".to_string(),
            ..Default::default()
        }
    }

    /// Set API credentials
    pub fn with_credentials(mut self, api_key: String, api_secret: String) -> Self {
        self.api_key = Some(api_key);
        self.api_secret = Some(api_secret);
        self
    }
}

/// Bybit API errors
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("API error: {code} - {message}")]
    ApiError { code: i32, message: String },

    #[error("JSON parsing error: {0}")]
    ParseError(#[from] serde_json::Error),

    #[error("Invalid response format")]
    InvalidResponse,

    #[error("Rate limit exceeded")]
    RateLimited,

    #[error("Authentication required")]
    AuthRequired,
}

/// Bybit REST API client
#[derive(Clone)]
pub struct BybitClient {
    config: BybitConfig,
    client: Client,
}

impl BybitClient {
    /// Create a new client with default configuration
    pub fn new() -> Self {
        Self::with_config(BybitConfig::default())
    }

    /// Create a new client with custom configuration
    pub fn with_config(config: BybitConfig) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self { config, client }
    }

    /// Generate request signature
    fn sign(&self, timestamp: u64, params: &str) -> Option<String> {
        let api_key = self.config.api_key.as_ref()?;
        let api_secret = self.config.api_secret.as_ref()?;

        let sign_str = format!("{}{}{}", timestamp, api_key, params);
        let mut mac = Hmac::<Sha256>::new_from_slice(api_secret.as_bytes()).ok()?;
        mac.update(sign_str.as_bytes());
        Some(hex::encode(mac.finalize().into_bytes()))
    }

    /// Get current timestamp in milliseconds
    fn timestamp_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    /// Fetch kline/candlestick data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
    /// * `limit` - Number of candles to fetch (max 1000)
    ///
    /// # Example
    /// ```rust,no_run
    /// use efficientnet_trading::api::BybitClient;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let client = BybitClient::new();
    ///     let candles = client.fetch_klines("BTCUSDT", "5", 100).await.unwrap();
    ///     println!("Fetched {} candles", candles.len());
    /// }
    /// ```
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>, BybitError> {
        let url = format!("{}/v5/market/kline", self.config.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?;

        let json: KlineResponse = response.json().await?;

        if json.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: json.ret_code,
                message: json.ret_msg,
            });
        }

        let candles = json
            .result
            .list
            .into_iter()
            .map(|k| k.into_candle())
            .collect();

        Ok(candles)
    }

    /// Fetch order book data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `limit` - Depth limit (1, 50, 200, 500)
    pub async fn fetch_orderbook(
        &self,
        symbol: &str,
        limit: usize,
    ) -> Result<OrderBook, BybitError> {
        let url = format!("{}/v5/market/orderbook", self.config.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?;

        let json: OrderBookResponse = response.json().await?;

        if json.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: json.ret_code,
                message: json.ret_msg,
            });
        }

        Ok(json.result.into_orderbook())
    }

    /// Fetch recent trades
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `limit` - Number of trades to fetch (max 1000)
    pub async fn fetch_trades(
        &self,
        symbol: &str,
        limit: usize,
    ) -> Result<Vec<Trade>, BybitError> {
        let url = format!("{}/v5/market/recent-trade", self.config.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?;

        let json: TradesResponse = response.json().await?;

        if json.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: json.ret_code,
                message: json.ret_msg,
            });
        }

        let trades = json
            .result
            .list
            .into_iter()
            .map(|t| t.into_trade())
            .collect();

        Ok(trades)
    }

    /// Fetch ticker information
    pub async fn fetch_ticker(&self, symbol: &str) -> Result<TickerInfo, BybitError> {
        let url = format!("{}/v5/market/tickers", self.config.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[("category", "linear"), ("symbol", symbol)])
            .send()
            .await?;

        let json: TickerResponse = response.json().await?;

        if json.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: json.ret_code,
                message: json.ret_msg,
            });
        }

        json.result
            .list
            .into_iter()
            .next()
            .ok_or(BybitError::InvalidResponse)
    }

    /// Fetch historical klines with pagination
    pub async fn fetch_historical_klines(
        &self,
        symbol: &str,
        interval: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<Candle>, BybitError> {
        let mut all_candles = Vec::new();
        let mut current_end = end_time.timestamp_millis() as u64;
        let start_ms = start_time.timestamp_millis() as u64;

        while current_end > start_ms {
            let url = format!("{}/v5/market/kline", self.config.base_url);

            let response = self
                .client
                .get(&url)
                .query(&[
                    ("category", "linear"),
                    ("symbol", symbol),
                    ("interval", interval),
                    ("limit", "1000"),
                    ("end", &current_end.to_string()),
                ])
                .send()
                .await?;

            let json: KlineResponse = response.json().await?;

            if json.ret_code != 0 {
                return Err(BybitError::ApiError {
                    code: json.ret_code,
                    message: json.ret_msg,
                });
            }

            if json.result.list.is_empty() {
                break;
            }

            let mut candles: Vec<Candle> = json
                .result
                .list
                .into_iter()
                .map(|k| k.into_candle())
                .filter(|c| c.timestamp >= start_ms)
                .collect();

            if let Some(oldest) = candles.last() {
                current_end = oldest.timestamp - 1;
            } else {
                break;
            }

            all_candles.append(&mut candles);

            // Rate limiting
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        // Sort by timestamp ascending
        all_candles.sort_by_key(|c| c.timestamp);
        Ok(all_candles)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// Response structures

#[derive(Debug, Deserialize)]
struct KlineResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: KlineResult,
}

#[derive(Debug, Deserialize)]
struct KlineResult {
    symbol: String,
    category: String,
    list: Vec<KlineData>,
}

#[derive(Debug, Deserialize)]
struct KlineData(
    String, // timestamp
    String, // open
    String, // high
    String, // low
    String, // close
    String, // volume
    String, // turnover
);

impl KlineData {
    fn into_candle(self) -> Candle {
        Candle {
            timestamp: self.0.parse().unwrap_or(0),
            open: self.1.parse().unwrap_or(0.0),
            high: self.2.parse().unwrap_or(0.0),
            low: self.3.parse().unwrap_or(0.0),
            close: self.4.parse().unwrap_or(0.0),
            volume: self.5.parse().unwrap_or(0.0),
        }
    }
}

#[derive(Debug, Deserialize)]
struct OrderBookResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: OrderBookResult,
}

#[derive(Debug, Deserialize)]
struct OrderBookResult {
    s: String, // symbol
    b: Vec<(String, String)>, // bids
    a: Vec<(String, String)>, // asks
    ts: u64,
    u: u64,
}

impl OrderBookResult {
    fn into_orderbook(self) -> OrderBook {
        OrderBook {
            symbol: self.s,
            timestamp: self.ts,
            bids: self
                .b
                .into_iter()
                .map(|(p, q)| OrderBookLevel {
                    price: p.parse().unwrap_or(0.0),
                    quantity: q.parse().unwrap_or(0.0),
                })
                .collect(),
            asks: self
                .a
                .into_iter()
                .map(|(p, q)| OrderBookLevel {
                    price: p.parse().unwrap_or(0.0),
                    quantity: q.parse().unwrap_or(0.0),
                })
                .collect(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct TradesResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: TradesResult,
}

#[derive(Debug, Deserialize)]
struct TradesResult {
    category: String,
    list: Vec<TradeData>,
}

#[derive(Debug, Deserialize)]
struct TradeData {
    #[serde(rename = "execId")]
    exec_id: String,
    symbol: String,
    price: String,
    size: String,
    side: String,
    time: String,
    #[serde(rename = "isBlockTrade")]
    is_block_trade: bool,
}

impl TradeData {
    fn into_trade(self) -> Trade {
        Trade {
            id: self.exec_id,
            symbol: self.symbol,
            price: self.price.parse().unwrap_or(0.0),
            quantity: self.size.parse().unwrap_or(0.0),
            side: if self.side == "Buy" {
                crate::data::TradeSide::Buy
            } else {
                crate::data::TradeSide::Sell
            },
            timestamp: self.time.parse().unwrap_or(0),
        }
    }
}

#[derive(Debug, Deserialize)]
struct TickerResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: TickerResult,
}

#[derive(Debug, Deserialize)]
struct TickerResult {
    category: String,
    list: Vec<TickerInfo>,
}

/// Ticker information
#[derive(Debug, Clone, Deserialize)]
pub struct TickerInfo {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "indexPrice")]
    pub index_price: String,
    #[serde(rename = "markPrice")]
    pub mark_price: String,
    #[serde(rename = "prevPrice24h")]
    pub prev_price_24h: String,
    #[serde(rename = "price24hPcnt")]
    pub price_24h_pcnt: String,
    #[serde(rename = "highPrice24h")]
    pub high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_price_24h: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    #[serde(rename = "turnover24h")]
    pub turnover_24h: String,
}

impl TickerInfo {
    /// Get last price as f64
    pub fn last_price_f64(&self) -> f64 {
        self.last_price.parse().unwrap_or(0.0)
    }

    /// Get 24h price change percentage as f64
    pub fn price_change_24h(&self) -> f64 {
        self.price_24h_pcnt.parse().unwrap_or(0.0) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = BybitConfig::default();
        assert_eq!(config.base_url, "https://api.bybit.com");
        assert!(config.api_key.is_none());
    }

    #[test]
    fn test_config_testnet() {
        let config = BybitConfig::testnet();
        assert!(config.base_url.contains("testnet"));
    }

    #[test]
    fn test_kline_data_parsing() {
        let data = KlineData(
            "1672502400000".to_string(),
            "16500.0".to_string(),
            "16600.0".to_string(),
            "16400.0".to_string(),
            "16550.0".to_string(),
            "1000.0".to_string(),
            "16500000.0".to_string(),
        );

        let candle = data.into_candle();
        assert_eq!(candle.timestamp, 1672502400000);
        assert_eq!(candle.open, 16500.0);
        assert_eq!(candle.high, 16600.0);
        assert_eq!(candle.low, 16400.0);
        assert_eq!(candle.close, 16550.0);
        assert_eq!(candle.volume, 1000.0);
    }
}
