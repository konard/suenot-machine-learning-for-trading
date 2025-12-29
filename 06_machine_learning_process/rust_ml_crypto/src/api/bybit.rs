//! Bybit exchange API client
//!
//! Provides methods to fetch market data from Bybit:
//! - Kline (candlestick) data
//! - Order book data
//! - Recent trades
//!
//! # Example
//!
//! ```rust,no_run
//! use ml_crypto::api::BybitClient;
//!
//! #[tokio::main]
//! async fn main() {
//!     let client = BybitClient::new();
//!     let candles = client.get_klines("BTCUSDT", "1h", 100).await.unwrap();
//!     println!("Got {} candles", candles.len());
//! }
//! ```

use super::error::{ApiError, ApiResult};
use crate::data::types::{Candle, OrderBook, OrderBookLevel, Trade};
use chrono::{DateTime, TimeZone, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Bybit API base URL
const BASE_URL: &str = "https://api.bybit.com";

/// Bybit API client for fetching market data
#[derive(Debug, Clone)]
pub struct BybitClient {
    client: Client,
    base_url: String,
}

/// Response wrapper from Bybit API
#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: T,
}

/// Kline result from Bybit API
#[derive(Debug, Deserialize)]
struct KlineResult {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

/// Order book result from Bybit API
#[derive(Debug, Deserialize)]
struct OrderBookResult {
    s: String, // symbol
    b: Vec<Vec<String>>, // bids
    a: Vec<Vec<String>>, // asks
    ts: u64, // timestamp
}

/// Trade result from Bybit API
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

/// Available kline intervals
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interval {
    Min1,
    Min3,
    Min5,
    Min15,
    Min30,
    Hour1,
    Hour2,
    Hour4,
    Hour6,
    Hour12,
    Day1,
    Week1,
    Month1,
}

impl Interval {
    /// Convert interval to API string
    pub fn as_str(&self) -> &'static str {
        match self {
            Interval::Min1 => "1",
            Interval::Min3 => "3",
            Interval::Min5 => "5",
            Interval::Min15 => "15",
            Interval::Min30 => "30",
            Interval::Hour1 => "60",
            Interval::Hour2 => "120",
            Interval::Hour4 => "240",
            Interval::Hour6 => "360",
            Interval::Hour12 => "720",
            Interval::Day1 => "D",
            Interval::Week1 => "W",
            Interval::Month1 => "M",
        }
    }

    /// Parse interval from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "1" | "1m" | "1min" => Some(Interval::Min1),
            "3" | "3m" | "3min" => Some(Interval::Min3),
            "5" | "5m" | "5min" => Some(Interval::Min5),
            "15" | "15m" | "15min" => Some(Interval::Min15),
            "30" | "30m" | "30min" => Some(Interval::Min30),
            "60" | "1h" | "1hour" => Some(Interval::Hour1),
            "120" | "2h" | "2hour" => Some(Interval::Hour2),
            "240" | "4h" | "4hour" => Some(Interval::Hour4),
            "360" | "6h" | "6hour" => Some(Interval::Hour6),
            "720" | "12h" | "12hour" => Some(Interval::Hour12),
            "d" | "1d" | "day" => Some(Interval::Day1),
            "w" | "1w" | "week" => Some(Interval::Week1),
            "m" | "1M" | "month" => Some(Interval::Month1),
            _ => None,
        }
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Create a new Bybit client with default settings
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: BASE_URL.to_string(),
        }
    }

    /// Create a new Bybit client with custom base URL (for testnet)
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Create a testnet client
    pub fn testnet() -> Self {
        Self::with_base_url("https://api-testnet.bybit.com")
    }

    /// Fetch kline (candlestick) data
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Kline interval (e.g., "1h", "4h", "1d")
    /// * `limit` - Number of candles to fetch (max 1000)
    ///
    /// # Returns
    ///
    /// Vector of candles sorted by time (oldest first)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> ApiResult<Vec<Candle>> {
        let interval_enum = Interval::from_str(interval)
            .ok_or_else(|| ApiError::InvalidInterval(interval.to_string()))?;

        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url,
            symbol.to_uppercase(),
            interval_enum.as_str(),
            limit.min(1000)
        );

        let response: BybitResponse<KlineResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(ApiError::ApiResponseError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let mut candles: Vec<Candle> = response
            .result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() >= 7 {
                    Some(Candle {
                        timestamp: item[0].parse().ok()?,
                        open: item[1].parse().ok()?,
                        high: item[2].parse().ok()?,
                        low: item[3].parse().ok()?,
                        close: item[4].parse().ok()?,
                        volume: item[5].parse().ok()?,
                        turnover: item[6].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by timestamp (oldest first)
        candles.sort_by_key(|c| c.timestamp);

        Ok(candles)
    }

    /// Fetch kline data with time range
    pub async fn get_klines_range(
        &self,
        symbol: &str,
        interval: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> ApiResult<Vec<Candle>> {
        let interval_enum = Interval::from_str(interval)
            .ok_or_else(|| ApiError::InvalidInterval(interval.to_string()))?;

        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&start={}&end={}&limit=1000",
            self.base_url,
            symbol.to_uppercase(),
            interval_enum.as_str(),
            start_time.timestamp_millis(),
            end_time.timestamp_millis()
        );

        let response: BybitResponse<KlineResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(ApiError::ApiResponseError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let mut candles: Vec<Candle> = response
            .result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() >= 7 {
                    Some(Candle {
                        timestamp: item[0].parse().ok()?,
                        open: item[1].parse().ok()?,
                        high: item[2].parse().ok()?,
                        low: item[3].parse().ok()?,
                        close: item[4].parse().ok()?,
                        volume: item[5].parse().ok()?,
                        turnover: item[6].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        candles.sort_by_key(|c| c.timestamp);
        Ok(candles)
    }

    /// Fetch order book data
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `limit` - Depth limit (1, 25, 50, 100, 200)
    pub async fn get_orderbook(&self, symbol: &str, limit: usize) -> ApiResult<OrderBook> {
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url,
            symbol.to_uppercase(),
            limit.min(200)
        );

        let response: BybitResponse<OrderBookResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(ApiError::ApiResponseError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

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

        Ok(OrderBook {
            symbol: response.result.s,
            timestamp: response.result.ts,
            bids: parse_levels(&response.result.b),
            asks: parse_levels(&response.result.a),
        })
    }

    /// Fetch recent trades
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `limit` - Number of trades to fetch (max 1000)
    pub async fn get_recent_trades(&self, symbol: &str, limit: usize) -> ApiResult<Vec<Trade>> {
        let url = format!(
            "{}/v5/market/recent-trade?category=spot&symbol={}&limit={}",
            self.base_url,
            symbol.to_uppercase(),
            limit.min(1000)
        );

        let response: BybitResponse<TradeResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(ApiError::ApiResponseError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let trades: Vec<Trade> = response
            .result
            .list
            .iter()
            .filter_map(|item| {
                Some(Trade {
                    id: item.exec_id.clone(),
                    symbol: item.symbol.clone(),
                    price: item.price.parse().ok()?,
                    quantity: item.size.parse().ok()?,
                    side: if item.side == "Buy" {
                        crate::data::types::TradeSide::Buy
                    } else {
                        crate::data::types::TradeSide::Sell
                    },
                    timestamp: item.time.parse().ok()?,
                })
            })
            .collect();

        Ok(trades)
    }

    /// Get available trading symbols
    pub async fn get_symbols(&self) -> ApiResult<Vec<String>> {
        #[derive(Deserialize)]
        struct SymbolResult {
            list: Vec<SymbolInfo>,
        }

        #[derive(Deserialize)]
        struct SymbolInfo {
            symbol: String,
        }

        let url = format!(
            "{}/v5/market/instruments-info?category=spot",
            self.base_url
        );

        let response: BybitResponse<SymbolResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(ApiError::ApiResponseError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        Ok(response.result.list.into_iter().map(|s| s.symbol).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_parsing() {
        assert_eq!(Interval::from_str("1h"), Some(Interval::Hour1));
        assert_eq!(Interval::from_str("4h"), Some(Interval::Hour4));
        assert_eq!(Interval::from_str("1d"), Some(Interval::Day1));
        assert_eq!(Interval::from_str("invalid"), None);
    }

    #[test]
    fn test_interval_as_str() {
        assert_eq!(Interval::Hour1.as_str(), "60");
        assert_eq!(Interval::Day1.as_str(), "D");
    }
}
