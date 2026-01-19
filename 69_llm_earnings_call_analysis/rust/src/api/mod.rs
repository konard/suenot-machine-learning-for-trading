//! Bybit API client module
//!
//! Provides access to Bybit cryptocurrency exchange data.

use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Bybit API base URL
const BYBIT_API_BASE: &str = "https://api.bybit.com";

/// Candle/OHLCV data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Ticker data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    pub symbol: String,
    pub last_price: f64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub volume_24h: f64,
    pub price_change_24h: f64,
}

/// Order book entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookEntry {
    pub price: f64,
    pub quantity: f64,
}

/// Order book data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub bids: Vec<OrderBookEntry>,
    pub asks: Vec<OrderBookEntry>,
    pub timestamp: i64,
}

/// Bybit API client
pub struct BybitClient {
    client: Client,
    base_url: String,
}

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: T,
}

/// Kline response data
#[derive(Debug, Deserialize)]
struct KlineResult {
    list: Vec<Vec<String>>,
}

/// Ticker response data
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
    bid_price: String,
    #[serde(rename = "ask1Price")]
    ask_price: String,
    #[serde(rename = "volume24h")]
    volume_24h: String,
    #[serde(rename = "price24hPcnt")]
    price_change_24h: String,
}

/// Order book response data
#[derive(Debug, Deserialize)]
struct OrderBookResult {
    b: Vec<Vec<String>>,
    a: Vec<Vec<String>>,
    ts: i64,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: BYBIT_API_BASE.to_string(),
        }
    }

    /// Create client with custom base URL (for testing)
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Get kline/candlestick data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Kline interval (e.g., "1", "5", "15", "60", "D")
    /// * `limit` - Number of candles to fetch (max 1000)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response: BybitResponse<KlineResult> =
            self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let candles = response
            .result
            .list
            .into_iter()
            .filter_map(|kline| {
                if kline.len() >= 6 {
                    Some(Candle {
                        timestamp: kline[0].parse().ok()?,
                        open: kline[1].parse().ok()?,
                        high: kline[2].parse().ok()?,
                        low: kline[3].parse().ok()?,
                        close: kline[4].parse().ok()?,
                        volume: kline[5].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(candles)
    }

    /// Get ticker information
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        let url = format!(
            "{}/v5/market/tickers?category=linear&symbol={}",
            self.base_url, symbol
        );

        let response: BybitResponse<TickerResult> =
            self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let ticker_data = response
            .result
            .list
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No ticker data found"))?;

        Ok(Ticker {
            symbol: ticker_data.symbol,
            last_price: ticker_data.last_price.parse()?,
            bid_price: ticker_data.bid_price.parse()?,
            ask_price: ticker_data.ask_price.parse()?,
            volume_24h: ticker_data.volume_24h.parse()?,
            price_change_24h: ticker_data.price_change_24h.parse::<f64>()? * 100.0,
        })
    }

    /// Get order book
    pub async fn get_orderbook(&self, symbol: &str, limit: usize) -> Result<OrderBook> {
        let url = format!(
            "{}/v5/market/orderbook?category=linear&symbol={}&limit={}",
            self.base_url, symbol, limit
        );

        let response: BybitResponse<OrderBookResult> =
            self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let bids = response
            .result
            .b
            .into_iter()
            .filter_map(|entry| {
                if entry.len() >= 2 {
                    Some(OrderBookEntry {
                        price: entry[0].parse().ok()?,
                        quantity: entry[1].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        let asks = response
            .result
            .a
            .into_iter()
            .filter_map(|entry| {
                if entry.len() >= 2 {
                    Some(OrderBookEntry {
                        price: entry[0].parse().ok()?,
                        quantity: entry[1].parse().ok()?,
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
        assert_eq!(client.base_url, BYBIT_API_BASE);
    }

    #[test]
    fn test_custom_base_url() {
        let client = BybitClient::with_base_url("https://test.api.com");
        assert_eq!(client.base_url, "https://test.api.com");
    }
}
