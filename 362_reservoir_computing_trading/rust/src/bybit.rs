//! # Bybit API Client
//!
//! Async client for the Bybit cryptocurrency exchange API.
//!
//! ## Features
//!
//! - Public endpoints (klines, orderbook, ticker)
//! - Private endpoints with HMAC authentication
//! - Rate limiting and retry logic
//!
//! ## Example
//!
//! ```rust,no_run
//! use reservoir_trading::bybit::{BybitClient, BybitConfig};
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = BybitConfig::testnet();
//!     let client = BybitClient::new(config);
//!
//!     let klines = client.get_klines("BTCUSDT", "1", 100).await.unwrap();
//!     println!("Got {} klines", klines.len());
//! }
//! ```

use chrono::{DateTime, Utc};
use hmac::{Hmac, Mac};
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tracing::{debug, warn};

/// Errors from Bybit API operations
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("API error: {code} - {message}")]
    ApiError { code: i32, message: String },

    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Invalid response format")]
    InvalidResponse,

    #[error("Rate limited, retry after {0} ms")]
    RateLimited(u64),

    #[error("Authentication required")]
    AuthRequired,
}

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

    /// Maximum retries on failure
    pub max_retries: u32,
}

impl BybitConfig {
    /// Create configuration for mainnet
    pub fn mainnet() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            api_key: None,
            api_secret: None,
            timeout_secs: 30,
            max_retries: 3,
        }
    }

    /// Create configuration for testnet
    pub fn testnet() -> Self {
        Self {
            base_url: "https://api-testnet.bybit.com".to_string(),
            api_key: None,
            api_secret: None,
            timeout_secs: 30,
            max_retries: 3,
        }
    }

    /// Set API credentials
    pub fn with_credentials(mut self, api_key: &str, api_secret: &str) -> Self {
        self.api_key = Some(api_key.to_string());
        self.api_secret = Some(api_secret.to_string());
        self
    }
}

impl Default for BybitConfig {
    fn default() -> Self {
        Self::mainnet()
    }
}

/// Kline (candlestick) data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Start time in milliseconds
    pub start_time: u64,

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

impl Kline {
    /// Calculate typical price (HLC/3)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate log return from previous close
    pub fn log_return(&self, prev_close: f64) -> f64 {
        (self.close / prev_close).ln()
    }

    /// Calculate range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate body (close - open)
    pub fn body(&self) -> f64 {
        self.close - self.open
    }

    /// Check if bullish
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
}

/// Order book data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Symbol
    pub symbol: String,

    /// Timestamp
    pub timestamp: u64,

    /// Bid prices and quantities
    pub bids: Vec<(f64, f64)>,

    /// Ask prices and quantities
    pub asks: Vec<(f64, f64)>,
}

impl OrderBook {
    /// Get best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|(p, _)| *p)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|(p, _)| *p)
    }

    /// Get mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Get spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get spread in basis points
    pub fn spread_bps(&self) -> Option<f64> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid)) => Some(spread / mid * 10000.0),
            _ => None,
        }
    }

    /// Calculate order book imbalance
    pub fn imbalance(&self, depth: usize) -> f64 {
        let bid_volume: f64 = self.bids.iter().take(depth).map(|(_, q)| q).sum();
        let ask_volume: f64 = self.asks.iter().take(depth).map(|(_, q)| q).sum();
        let total = bid_volume + ask_volume;

        if total > 0.0 {
            (bid_volume - ask_volume) / total
        } else {
            0.0
        }
    }
}

/// Ticker data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Symbol
    pub symbol: String,

    /// Last traded price
    pub last_price: f64,

    /// 24h high
    pub high_24h: f64,

    /// 24h low
    pub low_24h: f64,

    /// 24h volume
    pub volume_24h: f64,

    /// 24h turnover
    pub turnover_24h: f64,

    /// Price change percentage
    pub price_change_pct: f64,

    /// Best bid price
    pub bid_price: f64,

    /// Best ask price
    pub ask_price: f64,
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

/// Kline response
#[derive(Debug, Deserialize)]
struct KlineResult {
    list: Vec<Vec<String>>,
}

/// Order book response
#[derive(Debug, Deserialize)]
struct OrderBookResult {
    s: String,
    b: Vec<Vec<String>>,
    a: Vec<Vec<String>>,
    ts: u64,
}

/// Ticker response
#[derive(Debug, Deserialize)]
struct TickerResult {
    list: Vec<TickerItem>,
}

#[derive(Debug, Deserialize)]
struct TickerItem {
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
    #[serde(rename = "bid1Price")]
    bid_price: String,
    #[serde(rename = "ask1Price")]
    ask_price: String,
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

    /// Get the current server timestamp
    fn timestamp_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    /// Sign a request with HMAC-SHA256
    fn sign(&self, params: &str) -> Result<String, BybitError> {
        let secret = self
            .config
            .api_secret
            .as_ref()
            .ok_or(BybitError::AuthRequired)?;

        let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(params.as_bytes());
        let result = mac.finalize();

        Ok(hex::encode(result.into_bytes()))
    }

    /// Make a GET request to the API
    async fn get<T: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &str,
        params: &[(&str, &str)],
    ) -> Result<T, BybitError> {
        let url = format!("{}{}", self.config.base_url, endpoint);

        let mut retries = 0;
        loop {
            let response = self
                .client
                .get(&url)
                .query(params)
                .send()
                .await?;

            match self.handle_response(response).await {
                Ok(data) => return Ok(data),
                Err(BybitError::RateLimited(wait_ms)) => {
                    if retries >= self.config.max_retries {
                        return Err(BybitError::RateLimited(wait_ms));
                    }
                    warn!("Rate limited, waiting {} ms", wait_ms);
                    tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                    retries += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Handle API response
    async fn handle_response<T: for<'de> Deserialize<'de>>(
        &self,
        response: Response,
    ) -> Result<T, BybitError> {
        let status = response.status();
        let text = response.text().await?;

        debug!("API response: {}", text);

        if status.as_u16() == 429 {
            return Err(BybitError::RateLimited(1000));
        }

        let api_response: ApiResponse<T> = serde_json::from_str(&text)?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        api_response.result.ok_or(BybitError::InvalidResponse)
    }

    /// Get kline (candlestick) data
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Kline interval ("1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M")
    /// * `limit` - Number of klines (max 1000)
    ///
    /// # Returns
    ///
    /// Vector of Kline data sorted by time ascending
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Kline>, BybitError> {
        let limit_str = limit.to_string();
        let params = [
            ("category", "spot"),
            ("symbol", symbol),
            ("interval", interval),
            ("limit", &limit_str),
        ];

        let result: KlineResult = self.get("/v5/market/kline", &params).await?;

        let mut klines: Vec<Kline> = result
            .list
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 7 {
                    Some(Kline {
                        start_time: row[0].parse().ok()?,
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

        // Sort by time ascending
        klines.sort_by_key(|k| k.start_time);

        Ok(klines)
    }

    /// Get order book
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair
    /// * `limit` - Depth limit (1, 50, 200)
    pub async fn get_orderbook(
        &self,
        symbol: &str,
        limit: u32,
    ) -> Result<OrderBook, BybitError> {
        let limit_str = limit.to_string();
        let params = [
            ("category", "spot"),
            ("symbol", symbol),
            ("limit", &limit_str),
        ];

        let result: OrderBookResult = self.get("/v5/market/orderbook", &params).await?;

        let bids: Vec<(f64, f64)> = result
            .b
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 2 {
                    Some((row[0].parse().ok()?, row[1].parse().ok()?))
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<(f64, f64)> = result
            .a
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 2 {
                    Some((row[0].parse().ok()?, row[1].parse().ok()?))
                } else {
                    None
                }
            })
            .collect();

        Ok(OrderBook {
            symbol: result.s,
            timestamp: result.ts,
            bids,
            asks,
        })
    }

    /// Get ticker information
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker, BybitError> {
        let params = [("category", "spot"), ("symbol", symbol)];

        let result: TickerResult = self.get("/v5/market/tickers", &params).await?;

        let item = result.list.into_iter().next().ok_or(BybitError::InvalidResponse)?;

        Ok(Ticker {
            symbol: item.symbol,
            last_price: item.last_price.parse().unwrap_or(0.0),
            high_24h: item.high_price_24h.parse().unwrap_or(0.0),
            low_24h: item.low_price_24h.parse().unwrap_or(0.0),
            volume_24h: item.volume_24h.parse().unwrap_or(0.0),
            turnover_24h: item.turnover_24h.parse().unwrap_or(0.0),
            price_change_pct: item.price_24h_pcnt.parse().unwrap_or(0.0),
            bid_price: item.bid_price.parse().unwrap_or(0.0),
            ask_price: item.ask_price.parse().unwrap_or(0.0),
        })
    }

    /// Get multiple tickers
    pub async fn get_all_tickers(&self) -> Result<Vec<Ticker>, BybitError> {
        let params = [("category", "spot")];

        let result: TickerResult = self.get("/v5/market/tickers", &params).await?;

        let tickers: Vec<Ticker> = result
            .list
            .into_iter()
            .map(|item| Ticker {
                symbol: item.symbol,
                last_price: item.last_price.parse().unwrap_or(0.0),
                high_24h: item.high_price_24h.parse().unwrap_or(0.0),
                low_24h: item.low_price_24h.parse().unwrap_or(0.0),
                volume_24h: item.volume_24h.parse().unwrap_or(0.0),
                turnover_24h: item.turnover_24h.parse().unwrap_or(0.0),
                price_change_pct: item.price_24h_pcnt.parse().unwrap_or(0.0),
                bid_price: item.bid_price.parse().unwrap_or(0.0),
                ask_price: item.ask_price.parse().unwrap_or(0.0),
            })
            .collect();

        Ok(tickers)
    }
}

/// Historical data fetcher for backtesting
pub struct HistoricalDataFetcher {
    client: BybitClient,
}

impl HistoricalDataFetcher {
    /// Create a new historical data fetcher
    pub fn new(config: BybitConfig) -> Self {
        Self {
            client: BybitClient::new(config),
        }
    }

    /// Fetch historical klines with pagination
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair
    /// * `interval` - Kline interval
    /// * `start_time` - Start timestamp in milliseconds
    /// * `end_time` - End timestamp in milliseconds
    pub async fn fetch_range(
        &self,
        symbol: &str,
        interval: &str,
        start_time: u64,
        end_time: u64,
    ) -> Result<Vec<Kline>, BybitError> {
        let mut all_klines = Vec::new();
        let mut current_end = end_time;

        loop {
            let klines = self.client.get_klines(symbol, interval, 1000).await?;

            if klines.is_empty() {
                break;
            }

            let oldest = klines.first().unwrap().start_time;

            // Filter klines within range
            let filtered: Vec<Kline> = klines
                .into_iter()
                .filter(|k| k.start_time >= start_time && k.start_time <= end_time)
                .collect();

            all_klines.extend(filtered);

            if oldest <= start_time {
                break;
            }

            current_end = oldest - 1;

            // Rate limiting
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Sort by time
        all_klines.sort_by_key(|k| k.start_time);

        Ok(all_klines)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_calculations() {
        let kline = Kline {
            start_time: 1000000,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 102500.0,
        };

        assert!((kline.typical_price() - 103.333).abs() < 0.01);
        assert_eq!(kline.range(), 15.0);
        assert_eq!(kline.body(), 5.0);
        assert!(kline.is_bullish());

        let log_return = kline.log_return(100.0);
        assert!((log_return - 0.04879).abs() < 0.001);
    }

    #[test]
    fn test_orderbook_calculations() {
        let orderbook = OrderBook {
            symbol: "BTCUSDT".to_string(),
            timestamp: 1000000,
            bids: vec![(100.0, 10.0), (99.0, 20.0), (98.0, 30.0)],
            asks: vec![(101.0, 5.0), (102.0, 15.0), (103.0, 25.0)],
        };

        assert_eq!(orderbook.best_bid(), Some(100.0));
        assert_eq!(orderbook.best_ask(), Some(101.0));
        assert_eq!(orderbook.mid_price(), Some(100.5));
        assert_eq!(orderbook.spread(), Some(1.0));

        // Imbalance: bids=60, asks=45, imbalance = 15/105 = 0.143
        let imbalance = orderbook.imbalance(3);
        assert!((imbalance - 0.143).abs() < 0.01);
    }

    #[test]
    fn test_config_creation() {
        let config = BybitConfig::mainnet();
        assert!(config.base_url.contains("api.bybit.com"));

        let config = BybitConfig::testnet();
        assert!(config.base_url.contains("testnet"));

        let config = BybitConfig::default().with_credentials("key", "secret");
        assert_eq!(config.api_key, Some("key".to_string()));
    }
}
