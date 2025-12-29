//! # Bybit REST API Client
//!
//! Client for fetching order book and trade data from Bybit exchange.

use crate::data::orderbook::{OrderBook, OrderBookLevel};
use crate::data::trade::Trade;
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::endpoints;
use super::ApiError;

/// Bybit API client
#[derive(Debug, Clone)]
pub struct BybitClient {
    client: reqwest::Client,
    base_url: String,
    testnet: bool,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Create a new client for mainnet
    pub fn new() -> Self {
        Self::with_config(false)
    }

    /// Create a new client for testnet
    pub fn testnet() -> Self {
        Self::with_config(true)
    }

    /// Create a client with custom configuration
    pub fn with_config(testnet: bool) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("Failed to create HTTP client");

        let base_url = if testnet {
            endpoints::TESTNET_REST.to_string()
        } else {
            endpoints::MAINNET_REST.to_string()
        };

        Self {
            client,
            base_url,
            testnet,
        }
    }

    /// Check if using testnet
    pub fn is_testnet(&self) -> bool {
        self.testnet
    }

    /// Get order book for a symbol
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `limit` - Number of levels (1, 50, 200, 500)
    ///
    /// # Example
    /// ```rust,no_run
    /// use order_flow_imbalance::BybitClient;
    ///
    /// #[tokio::main]
    /// async fn main() -> anyhow::Result<()> {
    ///     let client = BybitClient::new();
    ///     let orderbook = client.get_orderbook("BTCUSDT", 50).await?;
    ///     println!("Best bid: {}", orderbook.best_bid().unwrap().price);
    ///     Ok(())
    /// }
    /// ```
    pub async fn get_orderbook(&self, symbol: &str, limit: u32) -> Result<OrderBook> {
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url, symbol, limit
        );

        let response = self.client.get(&url).send().await?;
        let data: BybitResponse<OrderBookResponse> = response.json().await?;

        if data.ret_code != 0 {
            return Err(ApiError::ApiResponse {
                code: data.ret_code,
                message: data.ret_msg,
            }
            .into());
        }

        let result = data.result.ok_or_else(|| {
            ApiError::ApiResponse {
                code: -1,
                message: "No data in response".to_string(),
            }
        })?;

        self.parse_orderbook(symbol, &result)
    }

    /// Get recent trades for a symbol
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `limit` - Number of trades (max 1000)
    pub async fn get_trades(&self, symbol: &str, limit: u32) -> Result<Vec<Trade>> {
        let limit = limit.min(1000);
        let url = format!(
            "{}/v5/market/recent-trade?category=spot&symbol={}&limit={}",
            self.base_url, symbol, limit
        );

        let response = self.client.get(&url).send().await?;
        let data: BybitResponse<TradesResponse> = response.json().await?;

        if data.ret_code != 0 {
            return Err(ApiError::ApiResponse {
                code: data.ret_code,
                message: data.ret_msg,
            }
            .into());
        }

        let result = data.result.ok_or_else(|| {
            ApiError::ApiResponse {
                code: -1,
                message: "No data in response".to_string(),
            }
        })?;

        self.parse_trades(symbol, &result)
    }

    /// Get kline/candlestick data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol
    /// * `interval` - Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
    /// * `limit` - Number of candles (max 1000)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Kline>> {
        let limit = limit.min(1000);
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response = self.client.get(&url).send().await?;
        let data: BybitResponse<KlinesResponse> = response.json().await?;

        if data.ret_code != 0 {
            return Err(ApiError::ApiResponse {
                code: data.ret_code,
                message: data.ret_msg,
            }
            .into());
        }

        let result = data.result.ok_or_else(|| {
            ApiError::ApiResponse {
                code: -1,
                message: "No data in response".to_string(),
            }
        })?;

        self.parse_klines(&result)
    }

    /// Get ticker information
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        let url = format!(
            "{}/v5/market/tickers?category=spot&symbol={}",
            self.base_url, symbol
        );

        let response = self.client.get(&url).send().await?;
        let data: BybitResponse<TickersResponse> = response.json().await?;

        if data.ret_code != 0 {
            return Err(ApiError::ApiResponse {
                code: data.ret_code,
                message: data.ret_msg,
            }
            .into());
        }

        let result = data.result.ok_or_else(|| {
            ApiError::ApiResponse {
                code: -1,
                message: "No data in response".to_string(),
            }
        })?;

        result
            .list
            .into_iter()
            .next()
            .ok_or_else(|| {
                ApiError::ApiResponse {
                    code: -1,
                    message: "No ticker data".to_string(),
                }
                .into()
            })
    }

    /// Parse order book from API response
    fn parse_orderbook(&self, symbol: &str, response: &OrderBookResponse) -> Result<OrderBook> {
        let timestamp = response
            .ts
            .parse::<i64>()
            .map(|ts| DateTime::from_timestamp_millis(ts).unwrap_or_else(Utc::now))
            .unwrap_or_else(|_| Utc::now());

        let bids: Vec<OrderBookLevel> = response
            .b
            .iter()
            .enumerate()
            .filter_map(|(i, level)| {
                let price = level.first()?.parse::<f64>().ok()?;
                let size = level.get(1)?.parse::<f64>().ok()?;
                Some(OrderBookLevel::new(price, size, i + 1))
            })
            .collect();

        let asks: Vec<OrderBookLevel> = response
            .a
            .iter()
            .enumerate()
            .filter_map(|(i, level)| {
                let price = level.first()?.parse::<f64>().ok()?;
                let size = level.get(1)?.parse::<f64>().ok()?;
                Some(OrderBookLevel::new(price, size, i + 1))
            })
            .collect();

        Ok(OrderBook::new(symbol.to_string(), timestamp, bids, asks))
    }

    /// Parse trades from API response
    fn parse_trades(&self, symbol: &str, response: &TradesResponse) -> Result<Vec<Trade>> {
        let trades: Vec<Trade> = response
            .list
            .iter()
            .filter_map(|t| {
                let timestamp = t
                    .time
                    .parse::<i64>()
                    .ok()
                    .and_then(DateTime::from_timestamp_millis)?;
                let price = t.price.parse::<f64>().ok()?;
                let size = t.size.parse::<f64>().ok()?;
                let is_buyer_maker = t.side == "Sell"; // Sell = buyer was taker

                Some(Trade::new(
                    symbol.to_string(),
                    timestamp,
                    price,
                    size,
                    is_buyer_maker,
                    t.exec_id.clone(),
                ))
            })
            .collect();

        Ok(trades)
    }

    /// Parse klines from API response
    fn parse_klines(&self, response: &KlinesResponse) -> Result<Vec<Kline>> {
        let klines: Vec<Kline> = response
            .list
            .iter()
            .filter_map(|k| {
                let timestamp = k
                    .first()?
                    .parse::<i64>()
                    .ok()
                    .and_then(DateTime::from_timestamp_millis)?;
                let open = k.get(1)?.parse::<f64>().ok()?;
                let high = k.get(2)?.parse::<f64>().ok()?;
                let low = k.get(3)?.parse::<f64>().ok()?;
                let close = k.get(4)?.parse::<f64>().ok()?;
                let volume = k.get(5)?.parse::<f64>().ok()?;
                let turnover = k.get(6)?.parse::<f64>().ok()?;

                Some(Kline {
                    timestamp,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    turnover,
                })
            })
            .collect();

        Ok(klines)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// API Response Types
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: Option<T>,
}

#[derive(Debug, Deserialize)]
struct OrderBookResponse {
    #[serde(rename = "s")]
    _symbol: String,
    #[serde(rename = "b")]
    b: Vec<Vec<String>>, // bids: [[price, size], ...]
    #[serde(rename = "a")]
    a: Vec<Vec<String>>, // asks: [[price, size], ...]
    #[serde(rename = "ts")]
    ts: String,
    #[serde(rename = "u")]
    _update_id: u64,
}

#[derive(Debug, Deserialize)]
struct TradesResponse {
    list: Vec<TradeData>,
}

#[derive(Debug, Deserialize)]
struct TradeData {
    #[serde(rename = "execId")]
    exec_id: String,
    price: String,
    size: String,
    side: String,
    time: String,
}

#[derive(Debug, Deserialize)]
struct KlinesResponse {
    list: Vec<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct TickersResponse {
    list: Vec<Ticker>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Public Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Kline/Candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub turnover: f64,
}

impl Kline {
    /// Get the midpoint price
    pub fn mid(&self) -> f64 {
        (self.high + self.low) / 2.0
    }

    /// Get the typical price
    pub fn typical(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Get the price range
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Check if bullish candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Get body size
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }
}

/// Ticker data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "highPrice24h")]
    pub high_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_24h: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    #[serde(rename = "turnover24h")]
    pub turnover_24h: String,
    #[serde(rename = "bid1Price")]
    pub best_bid: String,
    #[serde(rename = "ask1Price")]
    pub best_ask: String,
    #[serde(rename = "bid1Size")]
    pub bid_size: String,
    #[serde(rename = "ask1Size")]
    pub ask_size: String,
}

impl Ticker {
    /// Get the spread in price
    pub fn spread(&self) -> Option<f64> {
        let bid = self.best_bid.parse::<f64>().ok()?;
        let ask = self.best_ask.parse::<f64>().ok()?;
        Some(ask - bid)
    }

    /// Get the spread in basis points
    pub fn spread_bps(&self) -> Option<f64> {
        let bid = self.best_bid.parse::<f64>().ok()?;
        let ask = self.best_ask.parse::<f64>().ok()?;
        let mid = (bid + ask) / 2.0;
        Some((ask - bid) / mid * 10000.0)
    }

    /// Get the mid price
    pub fn mid_price(&self) -> Option<f64> {
        let bid = self.best_bid.parse::<f64>().ok()?;
        let ask = self.best_ask.parse::<f64>().ok()?;
        Some((bid + ask) / 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = BybitClient::new();
        assert!(!client.is_testnet());

        let testnet = BybitClient::testnet();
        assert!(testnet.is_testnet());
    }

    #[test]
    fn test_kline_methods() {
        let kline = Kline {
            timestamp: Utc::now(),
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        assert!(kline.is_bullish());
        assert_eq!(kline.body(), 5.0);
        assert_eq!(kline.range(), 15.0);
        assert_eq!(kline.mid(), 102.5);
    }
}
