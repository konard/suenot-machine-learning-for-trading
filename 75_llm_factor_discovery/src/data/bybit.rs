//! Bybit exchange client
//!
//! Provides integration with Bybit exchange for fetching market data
//! and executing trades.

use super::types::{MarketDataError, OhlcvBar, OrderBook, Ticker, TimeFrame};
use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Bybit-specific errors
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("API error: {code} - {message}")]
    ApiError { code: i32, message: String },

    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    #[error("Rate limit exceeded, retry after {0}ms")]
    RateLimitExceeded(u64),

    #[error("Authentication required")]
    AuthRequired,

    #[error("Market data error: {0}")]
    MarketDataError(#[from] MarketDataError),
}

/// Bybit client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitConfig {
    /// Base URL for the API
    #[serde(default = "default_base_url")]
    pub base_url: String,

    /// API key (optional for public endpoints)
    pub api_key: Option<String>,

    /// API secret (optional for public endpoints)
    pub api_secret: Option<String>,

    /// Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,

    /// Use testnet
    #[serde(default)]
    pub testnet: bool,

    /// Rate limit requests per second
    #[serde(default = "default_rate_limit")]
    pub rate_limit_per_sec: u32,
}

fn default_base_url() -> String {
    "https://api.bybit.com".to_string()
}

fn default_timeout() -> u64 {
    30
}

fn default_rate_limit() -> u32 {
    10
}

impl Default for BybitConfig {
    fn default() -> Self {
        Self {
            base_url: default_base_url(),
            api_key: None,
            api_secret: None,
            timeout_secs: default_timeout(),
            testnet: false,
            rate_limit_per_sec: default_rate_limit(),
        }
    }
}

impl BybitConfig {
    /// Create a new configuration for testnet
    pub fn testnet() -> Self {
        Self {
            base_url: "https://api-testnet.bybit.com".to_string(),
            testnet: true,
            ..Default::default()
        }
    }

    /// Create configuration with API keys
    pub fn with_keys(api_key: String, api_secret: String) -> Self {
        Self {
            api_key: Some(api_key),
            api_secret: Some(api_secret),
            ..Default::default()
        }
    }
}

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: Option<T>,
    time: Option<i64>,
}

/// Kline response data
#[derive(Debug, Deserialize)]
struct KlineResult {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

/// Ticker response data
#[derive(Debug, Deserialize)]
struct TickerResult {
    category: String,
    list: Vec<TickerData>,
}

#[derive(Debug, Deserialize)]
struct TickerData {
    symbol: String,
    #[serde(rename = "lastPrice")]
    last_price: String,
    #[serde(rename = "bid1Price")]
    bid1_price: String,
    #[serde(rename = "ask1Price")]
    ask1_price: String,
    #[serde(rename = "volume24h")]
    volume_24h: String,
    #[serde(rename = "highPrice24h")]
    high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    low_price_24h: String,
    #[serde(rename = "price24hPcnt")]
    price_24h_pcnt: String,
}

/// Orderbook response data
#[derive(Debug, Deserialize)]
struct OrderbookResult {
    s: String, // symbol
    b: Vec<Vec<String>>, // bids
    a: Vec<Vec<String>>, // asks
    ts: i64,
    u: i64,
}

/// Bybit exchange client
pub struct BybitClient {
    config: BybitConfig,
    client: Client,
}

impl BybitClient {
    /// Create a new Bybit client with configuration
    pub fn new(config: BybitConfig) -> Result<Self, BybitError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()?;

        Ok(Self { config, client })
    }

    /// Create a new client with default configuration
    pub fn default_client() -> Result<Self, BybitError> {
        Self::new(BybitConfig::default())
    }

    /// Fetch OHLCV data (klines)
    pub async fn get_klines(
        &self,
        symbol: &str,
        timeframe: TimeFrame,
        limit: Option<u32>,
        start_time: Option<DateTime<Utc>>,
        end_time: Option<DateTime<Utc>>,
    ) -> Result<Vec<OhlcvBar>, BybitError> {
        let limit = limit.unwrap_or(200).min(1000);

        let mut url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.config.base_url,
            symbol,
            timeframe.to_bybit_interval(),
            limit
        );

        if let Some(start) = start_time {
            url.push_str(&format!("&start={}", start.timestamp_millis()));
        }

        if let Some(end) = end_time {
            url.push_str(&format!("&end={}", end.timestamp_millis()));
        }

        debug!("Fetching klines from: {}", url);

        let response: BybitResponse<KlineResult> = self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let result = response
            .result
            .ok_or_else(|| BybitError::InvalidResponse("No result in response".to_string()))?;

        let bars: Vec<OhlcvBar> = result
            .list
            .into_iter()
            .filter_map(|kline| {
                if kline.len() < 6 {
                    return None;
                }
                Some(OhlcvBar {
                    timestamp: Utc.timestamp_millis_opt(kline[0].parse().ok()?).single()?,
                    open: kline[1].parse().ok()?,
                    high: kline[2].parse().ok()?,
                    low: kline[3].parse().ok()?,
                    close: kline[4].parse().ok()?,
                    volume: kline[5].parse().ok()?,
                    turnover: kline.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                })
            })
            .collect();

        // Bybit returns newest first, we want oldest first
        let mut bars = bars;
        bars.reverse();

        info!("Fetched {} klines for {}", bars.len(), symbol);

        Ok(bars)
    }

    /// Fetch current ticker data
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker, BybitError> {
        let url = format!(
            "{}/v5/market/tickers?category=linear&symbol={}",
            self.config.base_url, symbol
        );

        debug!("Fetching ticker from: {}", url);

        let response: BybitResponse<TickerResult> =
            self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let result = response
            .result
            .ok_or_else(|| BybitError::InvalidResponse("No result in response".to_string()))?;

        let ticker_data = result
            .list
            .into_iter()
            .next()
            .ok_or_else(|| BybitError::InvalidResponse("No ticker data".to_string()))?;

        let ticker = Ticker {
            symbol: ticker_data.symbol,
            last_price: ticker_data.last_price.parse().unwrap_or(0.0),
            bid_price: ticker_data.bid1_price.parse().unwrap_or(0.0),
            ask_price: ticker_data.ask1_price.parse().unwrap_or(0.0),
            volume_24h: ticker_data.volume_24h.parse().unwrap_or(0.0),
            high_24h: ticker_data.high_price_24h.parse().unwrap_or(0.0),
            low_24h: ticker_data.low_price_24h.parse().unwrap_or(0.0),
            price_change_24h: ticker_data.price_24h_pcnt.parse().unwrap_or(0.0),
            timestamp: Utc::now(),
        };

        Ok(ticker)
    }

    /// Fetch order book data
    pub async fn get_orderbook(&self, symbol: &str, limit: Option<u32>) -> Result<OrderBook, BybitError> {
        let limit = limit.unwrap_or(25).min(500);

        let url = format!(
            "{}/v5/market/orderbook?category=linear&symbol={}&limit={}",
            self.config.base_url, symbol, limit
        );

        debug!("Fetching orderbook from: {}", url);

        let response: BybitResponse<OrderbookResult> =
            self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let result = response
            .result
            .ok_or_else(|| BybitError::InvalidResponse("No result in response".to_string()))?;

        let bids: Vec<(f64, f64)> = result
            .b
            .into_iter()
            .filter_map(|level| {
                if level.len() < 2 {
                    return None;
                }
                Some((level[0].parse().ok()?, level[1].parse().ok()?))
            })
            .collect();

        let asks: Vec<(f64, f64)> = result
            .a
            .into_iter()
            .filter_map(|level| {
                if level.len() < 2 {
                    return None;
                }
                Some((level[0].parse().ok()?, level[1].parse().ok()?))
            })
            .collect();

        let orderbook = OrderBook {
            symbol: result.s,
            bids,
            asks,
            timestamp: Utc.timestamp_millis_opt(result.ts).single().unwrap_or_else(Utc::now),
        };

        Ok(orderbook)
    }

    /// Get list of available symbols
    pub async fn get_symbols(&self) -> Result<Vec<String>, BybitError> {
        let url = format!(
            "{}/v5/market/instruments-info?category=linear",
            self.config.base_url
        );

        #[derive(Deserialize)]
        struct InstrumentResult {
            list: Vec<InstrumentInfo>,
        }

        #[derive(Deserialize)]
        struct InstrumentInfo {
            symbol: String,
        }

        let response: BybitResponse<InstrumentResult> =
            self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let result = response
            .result
            .ok_or_else(|| BybitError::InvalidResponse("No result in response".to_string()))?;

        let symbols: Vec<String> = result.list.into_iter().map(|i| i.symbol).collect();

        info!("Found {} available symbols", symbols.len());

        Ok(symbols)
    }
}

/// Trait for data providers
#[async_trait]
pub trait DataProvider: Send + Sync {
    /// Fetch OHLCV bars
    async fn fetch_bars(
        &self,
        symbol: &str,
        timeframe: TimeFrame,
        limit: usize,
    ) -> Result<Vec<OhlcvBar>, MarketDataError>;

    /// Fetch current ticker
    async fn fetch_ticker(&self, symbol: &str) -> Result<Ticker, MarketDataError>;
}

#[async_trait]
impl DataProvider for BybitClient {
    async fn fetch_bars(
        &self,
        symbol: &str,
        timeframe: TimeFrame,
        limit: usize,
    ) -> Result<Vec<OhlcvBar>, MarketDataError> {
        self.get_klines(symbol, timeframe, Some(limit as u32), None, None)
            .await
            .map_err(|e| MarketDataError::FetchError(e.to_string()))
    }

    async fn fetch_ticker(&self, symbol: &str) -> Result<Ticker, MarketDataError> {
        self.get_ticker(symbol)
            .await
            .map_err(|e| MarketDataError::FetchError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = BybitConfig::default();
        assert_eq!(config.base_url, "https://api.bybit.com");
        assert!(!config.testnet);
    }

    #[test]
    fn test_config_testnet() {
        let config = BybitConfig::testnet();
        assert!(config.testnet);
        assert!(config.base_url.contains("testnet"));
    }

    #[test]
    fn test_timeframe_conversion() {
        assert_eq!(TimeFrame::Minute1.to_bybit_interval(), "1");
        assert_eq!(TimeFrame::Hour1.to_bybit_interval(), "60");
        assert_eq!(TimeFrame::Day1.to_bybit_interval(), "D");
    }
}
