//! Bybit API client for fetching cryptocurrency market data
//!
//! This module provides an async client for the Bybit public API.
//! Supports fetching OHLCV data, order books, and recent trades.

use super::types::{Candle, Interval, OrderBook, OrderBookLevel, Trade, TradeSide};
use chrono::{DateTime, TimeZone, Utc};
use reqwest::Client;
use serde::Deserialize;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Bybit API base URLs
pub mod endpoints {
    /// Main API endpoint
    pub const MAINNET: &str = "https://api.bybit.com";
    /// Testnet API endpoint
    pub const TESTNET: &str = "https://api-testnet.bybit.com";
}

/// Errors that can occur when interacting with Bybit API
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),

    #[error("API returned error: {code} - {message}")]
    ApiError { code: i32, message: String },

    #[error("Failed to parse response: {0}")]
    ParseError(String),

    #[error("Invalid parameters: {0}")]
    InvalidParams(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,
}

/// Bybit API client
#[derive(Debug, Clone)]
pub struct BybitClient {
    client: Client,
    base_url: String,
}

/// Generic API response wrapper
#[derive(Debug, Deserialize)]
struct ApiResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: Option<T>,
}

/// Kline (candlestick) response
#[derive(Debug, Deserialize)]
struct KlineResult {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

/// Order book response
#[derive(Debug, Deserialize)]
struct OrderBookResult {
    s: String,  // symbol
    b: Vec<Vec<String>>,  // bids
    a: Vec<Vec<String>>,  // asks
    ts: i64,  // timestamp
}

/// Recent trades response
#[derive(Debug, Deserialize)]
struct TradesResult {
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
}

impl BybitClient {
    /// Create a new Bybit client for mainnet
    pub fn new() -> Self {
        Self::with_base_url(endpoints::MAINNET)
    }

    /// Create a new Bybit client for testnet
    pub fn testnet() -> Self {
        Self::with_base_url(endpoints::TESTNET)
    }

    /// Create a new Bybit client with a custom base URL
    pub fn with_base_url(base_url: &str) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: base_url.to_string(),
        }
    }

    /// Fetch OHLCV (kline) data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Candle interval
    /// * `limit` - Number of candles to fetch (max 1000)
    /// * `start` - Optional start timestamp
    /// * `end` - Optional end timestamp
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: Interval,
        limit: Option<u32>,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Result<Vec<Candle>, BybitError> {
        let limit = limit.unwrap_or(200).min(1000);

        let mut url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url,
            symbol,
            interval.to_bybit_string(),
            limit
        );

        if let Some(start_time) = start {
            url.push_str(&format!("&start={}", start_time.timestamp_millis()));
        }

        if let Some(end_time) = end {
            url.push_str(&format!("&end={}", end_time.timestamp_millis()));
        }

        debug!("Fetching klines: {}", url);

        let response: ApiResponse<KlineResult> = self.client
            .get(&url)
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
            BybitError::ParseError("No result in response".to_string())
        })?;

        let mut candles: Vec<Candle> = result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() < 7 {
                    warn!("Skipping invalid kline data: {:?}", item);
                    return None;
                }

                let timestamp_ms: i64 = item[0].parse().ok()?;
                let timestamp = Utc.timestamp_millis_opt(timestamp_ms).single()?;

                Some(Candle {
                    timestamp,
                    symbol: symbol.to_string(),
                    open: item[1].parse().ok()?,
                    high: item[2].parse().ok()?,
                    low: item[3].parse().ok()?,
                    close: item[4].parse().ok()?,
                    volume: item[5].parse().ok()?,
                    turnover: item[6].parse().ok()?,
                })
            })
            .collect();

        // Bybit returns data in descending order, reverse to get chronological order
        candles.reverse();

        info!("Fetched {} candles for {}", candles.len(), symbol);

        Ok(candles)
    }

    /// Fetch historical klines with pagination (for larger datasets)
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol
    /// * `interval` - Candle interval
    /// * `start` - Start timestamp
    /// * `end` - End timestamp
    pub async fn get_historical_klines(
        &self,
        symbol: &str,
        interval: Interval,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<Candle>, BybitError> {
        let mut all_candles = Vec::new();
        let mut current_end = end;

        loop {
            let candles = self
                .get_klines(symbol, interval, Some(1000), Some(start), Some(current_end))
                .await?;

            if candles.is_empty() {
                break;
            }

            let earliest = candles.first().unwrap().timestamp;

            // Prepend new candles
            let mut new_candles = candles;
            new_candles.append(&mut all_candles);
            all_candles = new_candles;

            if earliest <= start {
                break;
            }

            // Move the end time window
            current_end = earliest;

            // Rate limiting - avoid hitting API limits
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        // Filter to exact time range
        all_candles.retain(|c| c.timestamp >= start && c.timestamp <= end);

        info!(
            "Fetched {} historical candles for {} from {} to {}",
            all_candles.len(),
            symbol,
            start,
            end
        );

        Ok(all_candles)
    }

    /// Fetch order book snapshot
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol
    /// * `limit` - Number of levels to fetch (max 200)
    pub async fn get_orderbook(
        &self,
        symbol: &str,
        limit: Option<u32>,
    ) -> Result<OrderBook, BybitError> {
        let limit = limit.unwrap_or(50).min(200);

        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url, symbol, limit
        );

        debug!("Fetching orderbook: {}", url);

        let response: ApiResponse<OrderBookResult> = self.client
            .get(&url)
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
            BybitError::ParseError("No result in response".to_string())
        })?;

        let parse_levels = |levels: &[Vec<String>]| -> Vec<OrderBookLevel> {
            levels
                .iter()
                .filter_map(|level| {
                    if level.len() >= 2 {
                        Some(OrderBookLevel {
                            price: level[0].parse().ok()?,
                            quantity: level[1].parse().ok()?,
                        })
                    } else {
                        None
                    }
                })
                .collect()
        };

        let timestamp = Utc.timestamp_millis_opt(result.ts).single()
            .unwrap_or_else(Utc::now);

        Ok(OrderBook {
            symbol: result.s,
            timestamp,
            bids: parse_levels(&result.b),
            asks: parse_levels(&result.a),
        })
    }

    /// Fetch recent trades
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol
    /// * `limit` - Number of trades to fetch (max 1000)
    pub async fn get_recent_trades(
        &self,
        symbol: &str,
        limit: Option<u32>,
    ) -> Result<Vec<Trade>, BybitError> {
        let limit = limit.unwrap_or(100).min(1000);

        let url = format!(
            "{}/v5/market/recent-trade?category=spot&symbol={}&limit={}",
            self.base_url, symbol, limit
        );

        debug!("Fetching recent trades: {}", url);

        let response: ApiResponse<TradesResult> = self.client
            .get(&url)
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
            BybitError::ParseError("No result in response".to_string())
        })?;

        let trades: Vec<Trade> = result
            .list
            .iter()
            .filter_map(|item| {
                let timestamp_ms: i64 = item.time.parse().ok()?;
                let timestamp = Utc.timestamp_millis_opt(timestamp_ms).single()?;

                Some(Trade {
                    id: item.exec_id.clone(),
                    symbol: item.symbol.clone(),
                    timestamp,
                    price: item.price.parse().ok()?,
                    quantity: item.size.parse().ok()?,
                    side: if item.side == "Buy" {
                        TradeSide::Buy
                    } else {
                        TradeSide::Sell
                    },
                })
            })
            .collect();

        info!("Fetched {} recent trades for {}", trades.len(), symbol);

        Ok(trades)
    }

    /// Get available trading symbols
    pub async fn get_symbols(&self) -> Result<Vec<String>, BybitError> {
        let url = format!(
            "{}/v5/market/instruments-info?category=spot",
            self.base_url
        );

        #[derive(Deserialize)]
        struct InstrumentsResult {
            list: Vec<InstrumentInfo>,
        }

        #[derive(Deserialize)]
        struct InstrumentInfo {
            symbol: String,
        }

        let response: ApiResponse<InstrumentsResult> = self.client
            .get(&url)
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
            BybitError::ParseError("No result in response".to_string())
        })?;

        Ok(result.list.into_iter().map(|i| i.symbol).collect())
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
    async fn test_get_klines() {
        let client = BybitClient::new();
        let candles = client
            .get_klines("BTCUSDT", Interval::Hour1, Some(10), None, None)
            .await;

        // Note: This test requires network access
        // In a real project, you'd mock the HTTP client
        if let Ok(candles) = candles {
            assert!(!candles.is_empty());
            assert_eq!(candles[0].symbol, "BTCUSDT");
        }
    }
}
