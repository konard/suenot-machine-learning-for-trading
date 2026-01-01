//! Bybit API Integration
//!
//! This module provides a client for fetching market data from Bybit exchange.
//! Supports both spot and derivatives (perpetual futures) markets.

use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur when interacting with Bybit API
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),

    #[error("API returned error: {code} - {message}")]
    ApiError { code: i32, message: String },

    #[error("Failed to parse response: {0}")]
    ParseError(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Invalid symbol: {0}")]
    InvalidSymbol(String),
}

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BybitResponse<T> {
    pub ret_code: i32,
    pub ret_msg: String,
    pub result: T,
    pub time: u64,
}

/// Kline (candlestick) data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Kline {
    /// Start timestamp in milliseconds
    pub start_time: u64,
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
    /// Turnover (quote volume)
    pub turnover: f64,
}

impl Kline {
    /// Parse from Bybit API response format
    pub fn from_api_data(data: &[String]) -> Result<Self, BybitError> {
        if data.len() < 7 {
            return Err(BybitError::ParseError("Invalid kline data".to_string()));
        }

        Ok(Self {
            start_time: data[0].parse().map_err(|_| {
                BybitError::ParseError("Invalid timestamp".to_string())
            })?,
            open: data[1].parse().map_err(|_| {
                BybitError::ParseError("Invalid open price".to_string())
            })?,
            high: data[2].parse().map_err(|_| {
                BybitError::ParseError("Invalid high price".to_string())
            })?,
            low: data[3].parse().map_err(|_| {
                BybitError::ParseError("Invalid low price".to_string())
            })?,
            close: data[4].parse().map_err(|_| {
                BybitError::ParseError("Invalid close price".to_string())
            })?,
            volume: data[5].parse().map_err(|_| {
                BybitError::ParseError("Invalid volume".to_string())
            })?,
            turnover: data[6].parse().map_err(|_| {
                BybitError::ParseError("Invalid turnover".to_string())
            })?,
        })
    }

    /// Get typical price (HLC/3)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Get true range
    pub fn true_range(&self, prev_close: Option<f64>) -> f64 {
        let hl = self.high - self.low;
        match prev_close {
            Some(pc) => {
                let hpc = (self.high - pc).abs();
                let lpc = (self.low - pc).abs();
                hl.max(hpc).max(lpc)
            }
            None => hl,
        }
    }

    /// Check if bullish candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Get body size
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Get upper shadow
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.close.max(self.open)
    }

    /// Get lower shadow
    pub fn lower_shadow(&self) -> f64 {
        self.close.min(self.open) - self.low
    }
}

/// Order book entry
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
}

/// Order book data
#[derive(Debug, Clone)]
pub struct OrderBook {
    /// Bid levels (buy orders)
    pub bids: Vec<OrderBookLevel>,
    /// Ask levels (sell orders)
    pub asks: Vec<OrderBookLevel>,
    /// Timestamp
    pub timestamp: u64,
}

impl OrderBook {
    /// Get best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Get mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Get bid-ask spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get spread as percentage
    pub fn spread_pct(&self) -> Option<f64> {
        match (self.mid_price(), self.spread()) {
            (Some(mid), Some(spread)) if mid > 0.0 => Some(spread / mid * 100.0),
            _ => None,
        }
    }

    /// Calculate order book imbalance
    pub fn imbalance(&self, depth: usize) -> f64 {
        let bid_volume: f64 = self.bids.iter().take(depth).map(|l| l.quantity).sum();
        let ask_volume: f64 = self.asks.iter().take(depth).map(|l| l.quantity).sum();

        let total = bid_volume + ask_volume;
        if total > 0.0 {
            (bid_volume - ask_volume) / total
        } else {
            0.0
        }
    }

    /// Calculate VWAP up to depth
    pub fn vwap(&self, depth: usize) -> (Option<f64>, Option<f64>) {
        let bid_vwap = Self::calculate_vwap(&self.bids, depth);
        let ask_vwap = Self::calculate_vwap(&self.asks, depth);
        (bid_vwap, ask_vwap)
    }

    fn calculate_vwap(levels: &[OrderBookLevel], depth: usize) -> Option<f64> {
        let subset: Vec<_> = levels.iter().take(depth).collect();
        if subset.is_empty() {
            return None;
        }

        let total_volume: f64 = subset.iter().map(|l| l.quantity).sum();
        if total_volume == 0.0 {
            return None;
        }

        let volume_price: f64 = subset.iter().map(|l| l.price * l.quantity).sum();
        Some(volume_price / total_volume)
    }
}

/// Kline API response format
#[derive(Debug, Deserialize)]
pub struct KlineResult {
    pub category: String,
    pub symbol: String,
    pub list: Vec<Vec<String>>,
}

/// Order book API response format
#[derive(Debug, Deserialize)]
pub struct OrderBookResult {
    pub s: String,  // symbol
    pub b: Vec<Vec<String>>,  // bids
    pub a: Vec<Vec<String>>,  // asks
    pub ts: u64,  // timestamp
    pub u: u64,  // update id
}

/// Bybit API client for market data
#[derive(Debug, Clone)]
pub struct BybitClient {
    /// Base URL for API
    base_url: String,
    /// HTTP client
    client: Client,
    /// Market category (spot, linear, inverse)
    category: String,
}

impl BybitClient {
    /// Create a new Bybit client for mainnet
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: Client::new(),
            category: "linear".to_string(),
        }
    }

    /// Create a client for testnet
    pub fn testnet() -> Self {
        Self {
            base_url: "https://api-testnet.bybit.com".to_string(),
            client: Client::new(),
            category: "linear".to_string(),
        }
    }

    /// Set market category (spot, linear, inverse)
    pub fn with_category(mut self, category: &str) -> Self {
        self.category = category.to_string();
        self
    }

    /// Fetch kline (candlestick) data
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
    /// * `limit` - Number of klines to fetch (max 1000)
    ///
    /// # Returns
    ///
    /// Vector of Kline data, sorted from oldest to newest
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>, BybitError> {
        let url = format!(
            "{}/v5/market/kline?category={}&symbol={}&interval={}&limit={}",
            self.base_url, self.category, symbol, interval, limit.min(1000)
        );

        let response: BybitResponse<KlineResult> = self
            .client
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

        let mut klines: Vec<Kline> = response
            .result
            .list
            .iter()
            .filter_map(|k| Kline::from_api_data(k).ok())
            .collect();

        // Bybit returns newest first, we want oldest first
        klines.reverse();

        Ok(klines)
    }

    /// Fetch order book data
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol
    /// * `limit` - Depth of order book (1, 25, 50, 100, 200)
    pub async fn get_orderbook(
        &self,
        symbol: &str,
        limit: usize,
    ) -> Result<OrderBook, BybitError> {
        let url = format!(
            "{}/v5/market/orderbook?category={}&symbol={}&limit={}",
            self.base_url, self.category, symbol, limit
        );

        let response: BybitResponse<OrderBookResult> = self
            .client
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

        let bids = response
            .result
            .b
            .iter()
            .filter_map(|l| {
                if l.len() >= 2 {
                    Some(OrderBookLevel {
                        price: l[0].parse().ok()?,
                        quantity: l[1].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        let asks = response
            .result
            .a
            .iter()
            .filter_map(|l| {
                if l.len() >= 2 {
                    Some(OrderBookLevel {
                        price: l[0].parse().ok()?,
                        quantity: l[1].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(OrderBook {
            bids,
            asks,
            timestamp: response.result.ts,
        })
    }

    /// Fetch ticker information
    pub async fn get_ticker(&self, symbol: &str) -> Result<TickerInfo, BybitError> {
        let url = format!(
            "{}/v5/market/tickers?category={}&symbol={}",
            self.base_url, self.category, symbol
        );

        let response: BybitResponse<TickerResult> = self
            .client
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

        response
            .result
            .list
            .into_iter()
            .next()
            .ok_or_else(|| BybitError::InvalidSymbol(symbol.to_string()))
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Ticker information
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TickerInfo {
    pub symbol: String,
    #[serde(deserialize_with = "deserialize_f64_from_str")]
    pub last_price: f64,
    #[serde(deserialize_with = "deserialize_f64_from_str")]
    pub index_price: f64,
    #[serde(deserialize_with = "deserialize_f64_from_str")]
    pub mark_price: f64,
    #[serde(deserialize_with = "deserialize_f64_from_str")]
    pub prev_price24h: f64,
    #[serde(deserialize_with = "deserialize_f64_from_str")]
    pub price24h_pcnt: f64,
    #[serde(deserialize_with = "deserialize_f64_from_str")]
    pub high_price24h: f64,
    #[serde(deserialize_with = "deserialize_f64_from_str")]
    pub low_price24h: f64,
    #[serde(deserialize_with = "deserialize_f64_from_str")]
    pub volume24h: f64,
    #[serde(deserialize_with = "deserialize_f64_from_str")]
    pub turnover24h: f64,
}

#[derive(Debug, Deserialize)]
pub struct TickerResult {
    pub list: Vec<TickerInfo>,
}

fn deserialize_f64_from_str<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.parse().map_err(serde::de::Error::custom)
}

/// Load sample data from CSV file
pub fn load_sample_data(path: &str) -> Result<Vec<Kline>, BybitError> {
    let mut reader = csv::Reader::from_path(path)
        .map_err(|e| BybitError::ParseError(e.to_string()))?;

    let mut klines = Vec::new();

    for result in reader.records() {
        let record = result.map_err(|e| BybitError::ParseError(e.to_string()))?;

        if record.len() >= 7 {
            let kline = Kline {
                start_time: record[0].parse().unwrap_or(0),
                open: record[1].parse().unwrap_or(0.0),
                high: record[2].parse().unwrap_or(0.0),
                low: record[3].parse().unwrap_or(0.0),
                close: record[4].parse().unwrap_or(0.0),
                volume: record[5].parse().unwrap_or(0.0),
                turnover: record[6].parse().unwrap_or(0.0),
            };
            klines.push(kline);
        }
    }

    Ok(klines)
}

/// Generate sample kline data for testing
pub fn generate_sample_data(n: usize, start_price: f64) -> Vec<Kline> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut klines = Vec::with_capacity(n);
    let mut price = start_price;
    let mut timestamp = 0u64;

    for _ in 0..n {
        let change = rng.gen_range(-0.02..0.02);
        let volatility = rng.gen_range(0.005..0.02);

        let open = price;
        let close = price * (1.0 + change);
        let high = open.max(close) * (1.0 + volatility);
        let low = open.min(close) * (1.0 - volatility);
        let volume = rng.gen_range(100.0..10000.0);

        klines.push(Kline {
            start_time: timestamp,
            open,
            high,
            low,
            close,
            volume,
            turnover: volume * (open + close) / 2.0,
        });

        price = close;
        timestamp += 60000; // 1 minute
    }

    klines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_properties() {
        let kline = Kline {
            start_time: 0,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        assert!(kline.is_bullish());
        assert_eq!(kline.body_size(), 5.0);
        assert_eq!(kline.typical_price(), (110.0 + 95.0 + 105.0) / 3.0);
    }

    #[test]
    fn test_orderbook_calculations() {
        let orderbook = OrderBook {
            bids: vec![
                OrderBookLevel { price: 100.0, quantity: 10.0 },
                OrderBookLevel { price: 99.0, quantity: 20.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 101.0, quantity: 15.0 },
                OrderBookLevel { price: 102.0, quantity: 25.0 },
            ],
            timestamp: 0,
        };

        assert_eq!(orderbook.best_bid(), Some(100.0));
        assert_eq!(orderbook.best_ask(), Some(101.0));
        assert_eq!(orderbook.mid_price(), Some(100.5));
        assert_eq!(orderbook.spread(), Some(1.0));
    }

    #[test]
    fn test_generate_sample_data() {
        let data = generate_sample_data(100, 50000.0);
        assert_eq!(data.len(), 100);

        for kline in &data {
            assert!(kline.high >= kline.low);
            assert!(kline.high >= kline.open);
            assert!(kline.high >= kline.close);
            assert!(kline.low <= kline.open);
            assert!(kline.low <= kline.close);
        }
    }
}
