//! Bybit exchange client for market data and order execution.

use crate::data::{
    MarketDataError, OhlcvBar, OrderBook, OrderBookLevel, OrderBookSnapshot, Ticker,
    TimeFrame, Trade, TradeDirection,
};
use chrono::{DateTime, TimeZone, Utc};
use hmac::{Hmac, Mac};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::HashMap;
use thiserror::Error;

/// Bybit-specific errors
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error: {code} - {message}")]
    Api { code: i32, message: String },

    #[error("Authentication error: {0}")]
    Auth(String),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Rate limit exceeded")]
    RateLimit,

    #[error("Insufficient balance")]
    InsufficientBalance,

    #[error("Order not found: {0}")]
    OrderNotFound(String),
}

impl From<BybitError> for MarketDataError {
    fn from(err: BybitError) -> Self {
        match err {
            BybitError::Http(e) => MarketDataError::Network(e),
            BybitError::Api { code, message } => MarketDataError::Exchange { code, message },
            BybitError::Auth(msg) => MarketDataError::Authentication(msg),
            BybitError::Parse(msg) => MarketDataError::Parse(msg),
            BybitError::RateLimit => MarketDataError::RateLimit,
            BybitError::InsufficientBalance => {
                MarketDataError::Exchange {
                    code: -1,
                    message: "Insufficient balance".to_string(),
                }
            }
            BybitError::OrderNotFound(id) => {
                MarketDataError::NotAvailable(format!("Order not found: {}", id))
            }
        }
    }
}

/// Bybit client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitConfig {
    /// API key (optional for public endpoints)
    pub api_key: Option<String>,
    /// API secret (optional for public endpoints)
    pub api_secret: Option<String>,
    /// Use testnet instead of mainnet
    pub testnet: bool,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
    /// Recv window for authenticated requests
    pub recv_window: u64,
}

impl Default for BybitConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            api_secret: None,
            testnet: false,
            timeout_ms: 5000,
            recv_window: 5000,
        }
    }
}

impl BybitConfig {
    /// Create a new config for mainnet with API credentials
    pub fn with_credentials(api_key: String, api_secret: String) -> Self {
        Self {
            api_key: Some(api_key),
            api_secret: Some(api_secret),
            ..Default::default()
        }
    }

    /// Create a new config for testnet
    pub fn testnet() -> Self {
        Self {
            testnet: true,
            ..Default::default()
        }
    }

    /// Create a new config for testnet with API credentials
    pub fn testnet_with_credentials(api_key: String, api_secret: String) -> Self {
        Self {
            api_key: Some(api_key),
            api_secret: Some(api_secret),
            testnet: true,
            ..Default::default()
        }
    }

    fn base_url(&self) -> &str {
        if self.testnet {
            "https://api-testnet.bybit.com"
        } else {
            "https://api.bybit.com"
        }
    }
}

/// Bybit REST API client
#[derive(Debug, Clone)]
pub struct BybitClient {
    config: BybitConfig,
    client: Client,
}

// API response structures
#[derive(Debug, Deserialize)]
struct ApiResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: Option<T>,
}

#[derive(Debug, Deserialize)]
struct KlineResult {
    list: Vec<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct TickerResult {
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
    #[serde(rename = "bid1Size")]
    bid1_size: String,
    #[serde(rename = "ask1Size")]
    ask1_size: String,
    #[serde(rename = "highPrice24h")]
    high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    low_price_24h: String,
    #[serde(rename = "volume24h")]
    volume_24h: String,
    #[serde(rename = "turnover24h")]
    turnover_24h: String,
    #[serde(rename = "openInterest", default)]
    open_interest: String,
    #[serde(rename = "fundingRate", default)]
    funding_rate: String,
    #[serde(rename = "nextFundingTime", default)]
    next_funding_time: String,
}

#[derive(Debug, Deserialize)]
struct OrderbookResult {
    s: String, // symbol
    b: Vec<Vec<String>>, // bids
    a: Vec<Vec<String>>, // asks
    ts: u64, // timestamp
    u: u64, // update id
}

#[derive(Debug, Deserialize)]
struct TradesResult {
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
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new(config: BybitConfig) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_millis(config.timeout_ms))
            .build()
            .expect("Failed to create HTTP client");

        Self { config, client }
    }

    /// Create a client for public endpoints only (no auth needed)
    pub fn public() -> Self {
        Self::new(BybitConfig::default())
    }

    /// Create a client for testnet public endpoints
    pub fn public_testnet() -> Self {
        Self::new(BybitConfig::testnet())
    }

    /// Generate signature for authenticated requests
    fn sign(&self, params: &str) -> Result<String, BybitError> {
        let secret = self
            .config
            .api_secret
            .as_ref()
            .ok_or_else(|| BybitError::Auth("API secret not configured".to_string()))?;

        let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes())
            .map_err(|e| BybitError::Auth(e.to_string()))?;

        mac.update(params.as_bytes());
        let result = mac.finalize();
        Ok(hex::encode(result.into_bytes()))
    }

    /// Add authentication headers to a request
    fn auth_headers(&self, params: &str) -> Result<HashMap<String, String>, BybitError> {
        let api_key = self
            .config
            .api_key
            .as_ref()
            .ok_or_else(|| BybitError::Auth("API key not configured".to_string()))?;

        let timestamp = Utc::now().timestamp_millis().to_string();
        let sign_str = format!(
            "{}{}{}{}",
            timestamp, api_key, self.config.recv_window, params
        );
        let signature = self.sign(&sign_str)?;

        let mut headers = HashMap::new();
        headers.insert("X-BAPI-API-KEY".to_string(), api_key.clone());
        headers.insert("X-BAPI-TIMESTAMP".to_string(), timestamp);
        headers.insert("X-BAPI-SIGN".to_string(), signature);
        headers.insert(
            "X-BAPI-RECV-WINDOW".to_string(),
            self.config.recv_window.to_string(),
        );

        Ok(headers)
    }

    /// Make a GET request to a public endpoint
    async fn get_public<T: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &str,
        params: &[(&str, &str)],
    ) -> Result<T, BybitError> {
        let url = format!("{}{}", self.config.base_url(), endpoint);

        let response = self
            .client
            .get(&url)
            .query(params)
            .send()
            .await?
            .json::<ApiResponse<T>>()
            .await?;

        if response.ret_code != 0 {
            if response.ret_code == 10006 {
                return Err(BybitError::RateLimit);
            }
            return Err(BybitError::Api {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        response
            .result
            .ok_or_else(|| BybitError::Parse("Empty result".to_string()))
    }

    /// Get OHLCV bars (klines)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: TimeFrame,
        limit: Option<u32>,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Result<Vec<OhlcvBar>, BybitError> {
        let mut params = vec![
            ("category", "linear"),
            ("symbol", symbol),
            ("interval", interval.to_bybit_interval()),
        ];

        let limit_str = limit.unwrap_or(200).to_string();
        params.push(("limit", &limit_str));

        let start_str;
        if let Some(s) = start {
            start_str = s.timestamp_millis().to_string();
            params.push(("start", &start_str));
        }

        let end_str;
        if let Some(e) = end {
            end_str = e.timestamp_millis().to_string();
            params.push(("end", &end_str));
        }

        let result: KlineResult = self.get_public("/v5/market/kline", &params).await?;

        let bars = result
            .list
            .into_iter()
            .filter_map(|row| {
                if row.len() < 6 {
                    return None;
                }

                let timestamp_ms: i64 = row[0].parse().ok()?;
                let timestamp = Utc.timestamp_millis_opt(timestamp_ms).single()?;

                Some(OhlcvBar {
                    timestamp,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
                    turnover: row.get(6).and_then(|s| s.parse().ok()),
                })
            })
            .collect();

        Ok(bars)
    }

    /// Get ticker for a symbol
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker, BybitError> {
        let params = [("category", "linear"), ("symbol", symbol)];

        let result: TickerResult = self.get_public("/v5/market/tickers", &params).await?;

        let data = result
            .list
            .into_iter()
            .next()
            .ok_or_else(|| BybitError::Parse("No ticker data".to_string()))?;

        Ok(Ticker {
            symbol: data.symbol,
            last_price: data.last_price.parse().unwrap_or(0.0),
            bid_price: data.bid1_price.parse().unwrap_or(0.0),
            ask_price: data.ask1_price.parse().unwrap_or(0.0),
            bid_qty: data.bid1_size.parse().unwrap_or(0.0),
            ask_qty: data.ask1_size.parse().unwrap_or(0.0),
            high_24h: data.high_price_24h.parse().unwrap_or(0.0),
            low_24h: data.low_price_24h.parse().unwrap_or(0.0),
            volume_24h: data.volume_24h.parse().unwrap_or(0.0),
            turnover_24h: data.turnover_24h.parse().unwrap_or(0.0),
            open_interest: data.open_interest.parse().ok(),
            funding_rate: data.funding_rate.parse().ok(),
            next_funding_time: data
                .next_funding_time
                .parse::<i64>()
                .ok()
                .and_then(|ts| Utc.timestamp_millis_opt(ts).single()),
            timestamp: Utc::now(),
        })
    }

    /// Get order book
    pub async fn get_orderbook(
        &self,
        symbol: &str,
        limit: Option<u32>,
    ) -> Result<OrderBook, BybitError> {
        let limit_str = limit.unwrap_or(50).to_string();
        let params = [
            ("category", "linear"),
            ("symbol", symbol),
            ("limit", &limit_str),
        ];

        let result: OrderbookResult = self.get_public("/v5/market/orderbook", &params).await?;

        let bids: Vec<OrderBookLevel> = result
            .b
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 2 {
                    Some(OrderBookLevel::new(
                        row[0].parse().ok()?,
                        row[1].parse().ok()?,
                    ))
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<OrderBookLevel> = result
            .a
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 2 {
                    Some(OrderBookLevel::new(
                        row[0].parse().ok()?,
                        row[1].parse().ok()?,
                    ))
                } else {
                    None
                }
            })
            .collect();

        let snapshot = OrderBookSnapshot {
            symbol: symbol.to_string(),
            bids,
            asks,
            timestamp: Utc.timestamp_millis_opt(result.ts as i64).single().unwrap_or_else(Utc::now),
            sequence: Some(result.u),
        };

        Ok(OrderBook::from_snapshot(snapshot))
    }

    /// Get recent trades
    pub async fn get_trades(
        &self,
        symbol: &str,
        limit: Option<u32>,
    ) -> Result<Vec<Trade>, BybitError> {
        let limit_str = limit.unwrap_or(500).to_string();
        let params = [
            ("category", "linear"),
            ("symbol", symbol),
            ("limit", &limit_str),
        ];

        let result: TradesResult = self.get_public("/v5/market/recent-trade", &params).await?;

        let trades = result
            .list
            .into_iter()
            .filter_map(|data| {
                let timestamp_ms: i64 = data.time.parse().ok()?;
                Some(Trade {
                    id: data.exec_id,
                    timestamp: Utc.timestamp_millis_opt(timestamp_ms).single()?,
                    price: data.price.parse().ok()?,
                    quantity: data.size.parse().ok()?,
                    direction: if data.side == "Buy" {
                        TradeDirection::Buy
                    } else {
                        TradeDirection::Sell
                    },
                })
            })
            .collect();

        Ok(trades)
    }

    /// Check if the client has authentication configured
    pub fn is_authenticated(&self) -> bool {
        self.config.api_key.is_some() && self.config.api_secret.is_some()
    }

    /// Get the base URL being used
    pub fn base_url(&self) -> &str {
        self.config.base_url()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = BybitConfig::default();
        assert!(config.api_key.is_none());
        assert!(!config.testnet);
        assert_eq!(config.base_url(), "https://api.bybit.com");
    }

    #[test]
    fn test_config_testnet() {
        let config = BybitConfig::testnet();
        assert!(config.testnet);
        assert_eq!(config.base_url(), "https://api-testnet.bybit.com");
    }

    #[test]
    fn test_client_public() {
        let client = BybitClient::public();
        assert!(!client.is_authenticated());
    }

    #[test]
    fn test_config_with_credentials() {
        let config = BybitConfig::with_credentials(
            "test_key".to_string(),
            "test_secret".to_string(),
        );
        assert!(config.api_key.is_some());
        assert!(config.api_secret.is_some());
        assert!(!config.testnet);
    }
}
