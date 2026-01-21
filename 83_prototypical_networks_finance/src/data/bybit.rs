//! Bybit API client for fetching market data
//!
//! This module provides an async client for interacting with the Bybit API
//! to fetch market data including klines, order books, trades, and more.

use crate::data::types::{Kline, OrderBook, OrderBookLevel, Ticker, Trade, FundingRate, OpenInterest};
use chrono::{DateTime, TimeZone, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Bybit API client configuration
#[derive(Debug, Clone)]
pub struct BybitConfig {
    /// Base URL for the API
    pub base_url: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Rate limit delay between requests in milliseconds
    pub rate_limit_ms: u64,
}

impl Default for BybitConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            timeout_secs: 30,
            rate_limit_ms: 100,
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
}

/// Errors that can occur when interacting with Bybit API
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("API error: {code} - {message}")]
    ApiError { code: i32, message: String },

    #[error("Invalid response format: {0}")]
    ParseError(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Invalid symbol: {0}")]
    InvalidSymbol(String),
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

/// Kline response from Bybit API
#[derive(Debug, Deserialize)]
struct KlineResult {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

/// Ticker response from Bybit API
#[derive(Debug, Deserialize)]
struct TickerResult {
    category: String,
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
    bid1_price: Option<String>,
    #[serde(rename = "ask1Price")]
    ask1_price: Option<String>,
}

/// Order book response from Bybit API
#[derive(Debug, Deserialize)]
struct OrderBookResult {
    s: String, // symbol
    b: Vec<Vec<String>>, // bids
    a: Vec<Vec<String>>, // asks
    ts: u64, // timestamp
}

/// Trade response from Bybit API
#[derive(Debug, Deserialize)]
struct TradeResult {
    category: String,
    list: Vec<TradeItem>,
}

#[derive(Debug, Deserialize)]
struct TradeItem {
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

/// Funding rate response
#[derive(Debug, Deserialize)]
struct FundingRateResult {
    category: String,
    list: Vec<FundingRateItem>,
}

#[derive(Debug, Deserialize)]
struct FundingRateItem {
    symbol: String,
    #[serde(rename = "fundingRate")]
    funding_rate: String,
    #[serde(rename = "fundingRateTimestamp")]
    funding_rate_timestamp: String,
}

/// Open interest response
#[derive(Debug, Deserialize)]
struct OpenInterestResult {
    category: String,
    list: Vec<OpenInterestItem>,
}

#[derive(Debug, Deserialize)]
struct OpenInterestItem {
    symbol: String,
    #[serde(rename = "openInterest")]
    open_interest: String,
    timestamp: String,
}

/// Bybit API client for fetching market data
pub struct BybitClient {
    config: BybitConfig,
    client: Client,
}

impl BybitClient {
    /// Create a new Bybit client with default configuration
    pub fn new() -> Result<Self, BybitError> {
        Self::with_config(BybitConfig::default())
    }

    /// Create a new Bybit client with custom configuration
    pub fn with_config(config: BybitConfig) -> Result<Self, BybitError> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()?;

        Ok(Self { config, client })
    }

    /// Fetch kline/candlestick data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Kline interval (e.g., "1", "5", "15", "60", "D")
    /// * `limit` - Number of klines to fetch (max 200)
    /// * `start_time` - Optional start time
    /// * `end_time` - Optional end time
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
        start_time: Option<DateTime<Utc>>,
        end_time: Option<DateTime<Utc>>,
    ) -> Result<Vec<Kline>, BybitError> {
        let mut params = vec![
            ("category", "linear".to_string()),
            ("symbol", symbol.to_string()),
            ("interval", interval.to_string()),
            ("limit", limit.min(200).to_string()),
        ];

        if let Some(start) = start_time {
            params.push(("start", (start.timestamp_millis()).to_string()));
        }
        if let Some(end) = end_time {
            params.push(("end", (end.timestamp_millis()).to_string()));
        }

        let url = format!("{}/v5/market/kline", self.config.base_url);
        let response: ApiResponse<KlineResult> = self.client
            .get(&url)
            .query(&params)
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

        let result = response.result.ok_or_else(|| {
            BybitError::ParseError("Missing result in response".to_string())
        })?;

        let klines = result.list
            .into_iter()
            .filter_map(|item| self.parse_kline(item))
            .collect();

        Ok(klines)
    }

    fn parse_kline(&self, item: Vec<String>) -> Option<Kline> {
        if item.len() < 7 {
            return None;
        }

        let timestamp_ms: i64 = item[0].parse().ok()?;
        let timestamp = Utc.timestamp_millis_opt(timestamp_ms).single()?;
        let open: f64 = item[1].parse().ok()?;
        let high: f64 = item[2].parse().ok()?;
        let low: f64 = item[3].parse().ok()?;
        let close: f64 = item[4].parse().ok()?;
        let volume: f64 = item[5].parse().ok()?;
        let quote_volume: f64 = item[6].parse().ok()?;

        Some(Kline {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            quote_volume,
            trade_count: None,
        })
    }

    /// Fetch ticker data for a symbol
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker, BybitError> {
        let url = format!("{}/v5/market/tickers", self.config.base_url);
        let response: ApiResponse<TickerResult> = self.client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
            ])
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

        let result = response.result.ok_or_else(|| {
            BybitError::ParseError("Missing result in response".to_string())
        })?;

        let item = result.list.into_iter().next().ok_or_else(|| {
            BybitError::InvalidSymbol(symbol.to_string())
        })?;

        Ok(Ticker {
            timestamp: Utc::now(),
            symbol: item.symbol,
            last_price: item.last_price.parse().unwrap_or(0.0),
            high_24h: item.high_price_24h.parse().unwrap_or(0.0),
            low_24h: item.low_price_24h.parse().unwrap_or(0.0),
            volume_24h: item.volume_24h.parse().unwrap_or(0.0),
            quote_volume_24h: item.turnover_24h.parse().unwrap_or(0.0),
            price_change_pct_24h: item.price_24h_pcnt.parse::<f64>().unwrap_or(0.0) * 100.0,
            bid_price: item.bid1_price.and_then(|p| p.parse().ok()),
            ask_price: item.ask1_price.and_then(|p| p.parse().ok()),
        })
    }

    /// Fetch order book data
    pub async fn get_order_book(
        &self,
        symbol: &str,
        limit: u32,
    ) -> Result<OrderBook, BybitError> {
        let url = format!("{}/v5/market/orderbook", self.config.base_url);
        let response: ApiResponse<OrderBookResult> = self.client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("limit", &limit.min(500).to_string()),
            ])
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

        let result = response.result.ok_or_else(|| {
            BybitError::ParseError("Missing result in response".to_string())
        })?;

        let bids = result.b
            .into_iter()
            .filter_map(|item| {
                if item.len() >= 2 {
                    Some(OrderBookLevel {
                        price: item[0].parse().ok()?,
                        quantity: item[1].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        let asks = result.a
            .into_iter()
            .filter_map(|item| {
                if item.len() >= 2 {
                    Some(OrderBookLevel {
                        price: item[0].parse().ok()?,
                        quantity: item[1].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(OrderBook {
            timestamp: Utc.timestamp_millis_opt(result.ts as i64).single().unwrap_or_else(Utc::now),
            symbol: result.s,
            bids,
            asks,
        })
    }

    /// Fetch recent trades
    pub async fn get_trades(
        &self,
        symbol: &str,
        limit: u32,
    ) -> Result<Vec<Trade>, BybitError> {
        let url = format!("{}/v5/market/recent-trade", self.config.base_url);
        let response: ApiResponse<TradeResult> = self.client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("limit", &limit.min(1000).to_string()),
            ])
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

        let result = response.result.ok_or_else(|| {
            BybitError::ParseError("Missing result in response".to_string())
        })?;

        let trades = result.list
            .into_iter()
            .filter_map(|item| {
                let timestamp_ms: i64 = item.time.parse().ok()?;
                Some(Trade {
                    id: item.exec_id,
                    timestamp: Utc.timestamp_millis_opt(timestamp_ms).single()?,
                    symbol: item.symbol,
                    price: item.price.parse().ok()?,
                    quantity: item.size.parse().ok()?,
                    is_buyer_maker: item.side == "Sell",
                })
            })
            .collect();

        Ok(trades)
    }

    /// Fetch funding rate for perpetual futures
    pub async fn get_funding_rate(&self, symbol: &str) -> Result<FundingRate, BybitError> {
        let url = format!("{}/v5/market/funding/history", self.config.base_url);
        let response: ApiResponse<FundingRateResult> = self.client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("limit", "1"),
            ])
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

        let result = response.result.ok_or_else(|| {
            BybitError::ParseError("Missing result in response".to_string())
        })?;

        let item = result.list.into_iter().next().ok_or_else(|| {
            BybitError::InvalidSymbol(symbol.to_string())
        })?;

        let timestamp_ms: i64 = item.funding_rate_timestamp.parse().unwrap_or(0);

        Ok(FundingRate {
            timestamp: Utc::now(),
            symbol: item.symbol,
            funding_rate: item.funding_rate.parse().unwrap_or(0.0),
            funding_time: Utc.timestamp_millis_opt(timestamp_ms).single().unwrap_or_else(Utc::now),
        })
    }

    /// Fetch open interest data
    pub async fn get_open_interest(&self, symbol: &str) -> Result<OpenInterest, BybitError> {
        let url = format!("{}/v5/market/open-interest", self.config.base_url);
        let response: ApiResponse<OpenInterestResult> = self.client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("intervalTime", "5min"),
                ("limit", "1"),
            ])
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

        let result = response.result.ok_or_else(|| {
            BybitError::ParseError("Missing result in response".to_string())
        })?;

        let item = result.list.into_iter().next().ok_or_else(|| {
            BybitError::InvalidSymbol(symbol.to_string())
        })?;

        let timestamp_ms: i64 = item.timestamp.parse().unwrap_or(0);
        let oi: f64 = item.open_interest.parse().unwrap_or(0.0);

        Ok(OpenInterest {
            timestamp: Utc.timestamp_millis_opt(timestamp_ms).single().unwrap_or_else(Utc::now),
            symbol: symbol.to_string(),
            open_interest: oi,
            open_interest_value: 0.0, // Would need price to calculate
        })
    }

    /// Fetch multiple data types for comprehensive market snapshot
    pub async fn get_market_snapshot(
        &self,
        symbol: &str,
    ) -> Result<MarketSnapshot, BybitError> {
        // Fetch all data concurrently
        let (ticker, order_book, trades, funding_rate, open_interest) = tokio::try_join!(
            self.get_ticker(symbol),
            self.get_order_book(symbol, 25),
            self.get_trades(symbol, 100),
            self.get_funding_rate(symbol),
            self.get_open_interest(symbol),
        )?;

        Ok(MarketSnapshot {
            symbol: symbol.to_string(),
            timestamp: Utc::now(),
            ticker,
            order_book,
            recent_trades: trades,
            funding_rate,
            open_interest,
        })
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new().expect("Failed to create default BybitClient")
    }
}

/// Complete market snapshot with all relevant data
#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub ticker: Ticker,
    pub order_book: OrderBook,
    pub recent_trades: Vec<Trade>,
    pub funding_rate: FundingRate,
    pub open_interest: OpenInterest,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bybit_config_default() {
        let config = BybitConfig::default();
        assert_eq!(config.base_url, "https://api.bybit.com");
        assert_eq!(config.timeout_secs, 30);
    }

    #[test]
    fn test_bybit_config_testnet() {
        let config = BybitConfig::testnet();
        assert!(config.base_url.contains("testnet"));
    }
}
