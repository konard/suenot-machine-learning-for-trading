//! Bybit API client for cryptocurrency market data.

use crate::data::{Candle, OrderBook, OrderBookLevel, Ticker, TimeFrame};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Bybit API base URL
const BYBIT_API_URL: &str = "https://api.bybit.com";

/// Errors from Bybit API
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("API error: {code} - {message}")]
    Api { code: i32, message: String },
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Rate limit exceeded")]
    RateLimit,
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
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

/// Ticker response
#[derive(Debug, Deserialize)]
struct TickerResult {
    list: Vec<TickerData>,
}

#[derive(Debug, Deserialize)]
struct TickerData {
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
    bid1_price: String,
    #[serde(rename = "ask1Price")]
    ask1_price: String,
}

/// Order book response
#[derive(Debug, Deserialize)]
struct OrderBookResult {
    s: String, // symbol
    b: Vec<Vec<String>>, // bids
    a: Vec<Vec<String>>, // asks
    ts: i64, // timestamp
}

/// Bybit API client
#[derive(Debug, Clone)]
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: BYBIT_API_URL.to_string(),
        }
    }

    /// Create client with custom base URL (for testing)
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
        }
    }

    /// Fetch candles for a symbol
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Time interval (e.g., "1h", "4h", "1d")
    /// * `limit` - Number of candles to fetch (max 200)
    pub async fn fetch_candles(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>, BybitError> {
        let timeframe = TimeFrame::from_str(interval).unwrap_or(TimeFrame::H1);
        let limit = limit.min(200);

        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url,
            symbol,
            timeframe.to_bybit_interval(),
            limit
        );

        let response: ApiResponse<KlineResult> = self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(BybitError::Api {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let result = response.result.ok_or_else(|| BybitError::Parse("No result".to_string()))?;

        let candles: Vec<Candle> = result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() >= 6 {
                    Some(Candle {
                        timestamp: item[0].parse().unwrap_or(0) / 1000, // Convert ms to s
                        open: item[1].parse().unwrap_or(0.0),
                        high: item[2].parse().unwrap_or(0.0),
                        low: item[3].parse().unwrap_or(0.0),
                        close: item[4].parse().unwrap_or(0.0),
                        volume: item[5].parse().unwrap_or(0.0),
                        turnover: if item.len() > 6 {
                            item[6].parse().unwrap_or(0.0)
                        } else {
                            0.0
                        },
                    })
                } else {
                    None
                }
            })
            .rev() // Bybit returns newest first, we want oldest first
            .collect();

        Ok(candles)
    }

    /// Fetch candles for multiple symbols
    pub async fn fetch_candles_multi(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: usize,
    ) -> Result<HashMap<String, Vec<Candle>>, BybitError> {
        let mut result = HashMap::new();

        for symbol in symbols {
            match self.fetch_candles(symbol, interval, limit).await {
                Ok(candles) => {
                    result.insert(symbol.to_string(), candles);
                }
                Err(e) => {
                    log::warn!("Failed to fetch candles for {}: {}", symbol, e);
                }
            }
        }

        Ok(result)
    }

    /// Fetch current ticker for a symbol
    pub async fn fetch_ticker(&self, symbol: &str) -> Result<Ticker, BybitError> {
        let url = format!(
            "{}/v5/market/tickers?category=spot&symbol={}",
            self.base_url, symbol
        );

        let response: ApiResponse<TickerResult> = self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(BybitError::Api {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let result = response.result.ok_or_else(|| BybitError::Parse("No result".to_string()))?;
        let data = result
            .list
            .first()
            .ok_or_else(|| BybitError::Parse("Empty list".to_string()))?;

        Ok(Ticker {
            symbol: data.symbol.clone(),
            last_price: data.last_price.parse().unwrap_or(0.0),
            high_24h: data.high_price_24h.parse().unwrap_or(0.0),
            low_24h: data.low_price_24h.parse().unwrap_or(0.0),
            volume_24h: data.volume_24h.parse().unwrap_or(0.0),
            turnover_24h: data.turnover_24h.parse().unwrap_or(0.0),
            price_change_24h: data.price_24h_pcnt.parse().unwrap_or(0.0) * 100.0,
            bid_price: data.bid1_price.parse().unwrap_or(0.0),
            ask_price: data.ask1_price.parse().unwrap_or(0.0),
            timestamp: chrono::Utc::now().timestamp(),
        })
    }

    /// Fetch tickers for multiple symbols
    pub async fn fetch_tickers(&self, symbols: &[&str]) -> Result<Vec<Ticker>, BybitError> {
        let mut tickers = Vec::new();

        for symbol in symbols {
            match self.fetch_ticker(symbol).await {
                Ok(ticker) => tickers.push(ticker),
                Err(e) => log::warn!("Failed to fetch ticker for {}: {}", symbol, e),
            }
        }

        Ok(tickers)
    }

    /// Fetch order book for a symbol
    pub async fn fetch_orderbook(
        &self,
        symbol: &str,
        depth: usize,
    ) -> Result<OrderBook, BybitError> {
        let depth = depth.min(200);
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url, symbol, depth
        );

        let response: ApiResponse<OrderBookResult> =
            self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(BybitError::Api {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let result = response.result.ok_or_else(|| BybitError::Parse("No result".to_string()))?;

        let bids: Vec<OrderBookLevel> = result
            .b
            .iter()
            .filter_map(|level| {
                if level.len() >= 2 {
                    Some(OrderBookLevel {
                        price: level[0].parse().unwrap_or(0.0),
                        quantity: level[1].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<OrderBookLevel> = result
            .a
            .iter()
            .filter_map(|level| {
                if level.len() >= 2 {
                    Some(OrderBookLevel {
                        price: level[0].parse().unwrap_or(0.0),
                        quantity: level[1].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(OrderBook {
            symbol: result.s,
            bids,
            asks,
            timestamp: result.ts / 1000, // Convert ms to s
        })
    }

    /// Get list of popular trading pairs
    pub fn popular_pairs() -> Vec<&'static str> {
        vec![
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
            "LINKUSDT", "LTCUSDT", "UNIUSDT", "ATOMUSDT", "NEARUSDT",
            "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT", "SEIUSDT",
        ]
    }

    /// Get list of DeFi tokens
    pub fn defi_pairs() -> Vec<&'static str> {
        vec![
            "UNIUSDT", "AAVEUSDT", "MKRUSDT", "COMPUSDT", "SNXUSDT",
            "CRVUSDT", "YFIUSDT", "SUSHIUSDT", "1INCHUSDT", "LDOUSDT",
        ]
    }

    /// Get list of Layer 1 tokens
    pub fn layer1_pairs() -> Vec<&'static str> {
        vec![
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "ADAUSDT",
            "DOTUSDT", "ATOMUSDT", "NEARUSDT", "APTUSDT", "SUIUSDT",
        ]
    }

    /// Get list of Layer 2 tokens
    pub fn layer2_pairs() -> Vec<&'static str> {
        vec![
            "MATICUSDT", "ARBUSDT", "OPUSDT", "IMXUSDT", "MANTAUSDT",
        ]
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

    #[test]
    fn test_client_creation() {
        let client = BybitClient::new();
        assert_eq!(client.base_url, BYBIT_API_URL);
    }

    #[test]
    fn test_popular_pairs() {
        let pairs = BybitClient::popular_pairs();
        assert!(pairs.contains(&"BTCUSDT"));
        assert!(pairs.contains(&"ETHUSDT"));
    }

    #[test]
    fn test_layer1_pairs() {
        let pairs = BybitClient::layer1_pairs();
        assert!(pairs.len() >= 5);
    }
}
