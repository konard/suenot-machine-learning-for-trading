//! Bybit exchange API client.
//!
//! Fetches market data from Bybit's public API.

use super::{Candle, OrderBook, Trade};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur when fetching data from Bybit.
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Rate limited")]
    RateLimited,
}

/// Bybit API response wrapper.
#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: T,
}

/// Kline data from Bybit API.
#[derive(Debug, Deserialize)]
struct KlineResult {
    list: Vec<Vec<String>>,
}

/// Order book data from Bybit API.
#[derive(Debug, Deserialize)]
struct OrderBookResult {
    s: String,          // symbol
    b: Vec<Vec<String>>, // bids
    a: Vec<Vec<String>>, // asks
    ts: u64,            // timestamp
}

/// Trade data from Bybit API.
#[derive(Debug, Deserialize)]
struct TradeResult {
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

/// Bybit API client for fetching market data.
#[derive(Clone)]
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client.
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Create a client with a custom base URL (for testnet).
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
        }
    }

    /// Create a testnet client.
    pub fn testnet() -> Self {
        Self::with_base_url("https://api-testnet.bybit.com")
    }

    /// Fetch kline (candlestick) data for a symbol.
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>, BybitError> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response: BybitResponse<KlineResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError(response.ret_msg));
        }

        let candles = response
            .result
            .list
            .into_iter()
            .filter_map(|row| {
                if row.len() < 6 {
                    return None;
                }
                Some(Candle {
                    timestamp: row[0].parse().ok()?,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
                    symbol: symbol.to_string(),
                })
            })
            .collect();

        Ok(candles)
    }

    /// Fetch candles for multiple symbols.
    pub async fn fetch_candles(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: usize,
    ) -> Result<HashMap<String, Vec<Candle>>, BybitError> {
        let mut result = HashMap::new();

        for symbol in symbols {
            match self.fetch_klines(symbol, interval, limit).await {
                Ok(mut candles) => {
                    // Reverse to get chronological order (Bybit returns newest first)
                    candles.reverse();
                    result.insert(symbol.to_string(), candles);
                }
                Err(e) => {
                    tracing::warn!("Failed to fetch candles for {}: {}", symbol, e);
                }
            }
        }

        Ok(result)
    }

    /// Fetch order book for a symbol.
    pub async fn fetch_orderbook(
        &self,
        symbol: &str,
        depth: usize,
    ) -> Result<OrderBook, BybitError> {
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url, symbol, depth
        );

        let response: BybitResponse<OrderBookResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError(response.ret_msg));
        }

        let bids: Vec<(f64, f64)> = response
            .result
            .b
            .into_iter()
            .filter_map(|row| {
                if row.len() < 2 {
                    return None;
                }
                Some((row[0].parse().ok()?, row[1].parse().ok()?))
            })
            .collect();

        let asks: Vec<(f64, f64)> = response
            .result
            .a
            .into_iter()
            .filter_map(|row| {
                if row.len() < 2 {
                    return None;
                }
                Some((row[0].parse().ok()?, row[1].parse().ok()?))
            })
            .collect();

        Ok(OrderBook {
            timestamp: response.result.ts,
            symbol: response.result.s,
            bids,
            asks,
        })
    }

    /// Fetch recent trades for a symbol.
    pub async fn fetch_trades(
        &self,
        symbol: &str,
        limit: usize,
    ) -> Result<Vec<Trade>, BybitError> {
        let url = format!(
            "{}/v5/market/recent-trade?category=spot&symbol={}&limit={}",
            self.base_url, symbol, limit
        );

        let response: BybitResponse<TradeResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError(response.ret_msg));
        }

        let trades = response
            .result
            .list
            .into_iter()
            .filter_map(|item| {
                Some(Trade {
                    timestamp: item.time.parse().ok()?,
                    symbol: item.symbol,
                    price: item.price.parse().ok()?,
                    quantity: item.size.parse().ok()?,
                    is_buyer_maker: item.side == "Sell",
                })
            })
            .collect();

        Ok(trades)
    }

    /// Fetch ticker information for a symbol.
    pub async fn fetch_ticker(&self, symbol: &str) -> Result<TickerInfo, BybitError> {
        let url = format!(
            "{}/v5/market/tickers?category=spot&symbol={}",
            self.base_url, symbol
        );

        let response: BybitResponse<TickerResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError(response.ret_msg));
        }

        let ticker = response
            .result
            .list
            .into_iter()
            .next()
            .ok_or_else(|| BybitError::ParseError("No ticker data".to_string()))?;

        Ok(TickerInfo {
            symbol: ticker.symbol,
            last_price: ticker.last_price.parse().unwrap_or(0.0),
            price_24h_pct: ticker.price_24h_pct_chg.parse().unwrap_or(0.0),
            high_24h: ticker.high_price_24h.parse().unwrap_or(0.0),
            low_24h: ticker.low_price_24h.parse().unwrap_or(0.0),
            volume_24h: ticker.volume_24h.parse().unwrap_or(0.0),
            turnover_24h: ticker.turnover_24h.parse().unwrap_or(0.0),
        })
    }

    /// Fetch tickers for all spot symbols.
    pub async fn fetch_all_tickers(&self) -> Result<Vec<TickerInfo>, BybitError> {
        let url = format!("{}/v5/market/tickers?category=spot", self.base_url);

        let response: BybitResponse<TickerResult> = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError(response.ret_msg));
        }

        let tickers = response
            .result
            .list
            .into_iter()
            .map(|ticker| TickerInfo {
                symbol: ticker.symbol,
                last_price: ticker.last_price.parse().unwrap_or(0.0),
                price_24h_pct: ticker.price_24h_pct_chg.parse().unwrap_or(0.0),
                high_24h: ticker.high_price_24h.parse().unwrap_or(0.0),
                low_24h: ticker.low_price_24h.parse().unwrap_or(0.0),
                volume_24h: ticker.volume_24h.parse().unwrap_or(0.0),
                turnover_24h: ticker.turnover_24h.parse().unwrap_or(0.0),
            })
            .collect();

        Ok(tickers)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Ticker result from API.
#[derive(Debug, Deserialize)]
struct TickerResult {
    list: Vec<TickerItem>,
}

#[derive(Debug, Deserialize)]
struct TickerItem {
    symbol: String,
    #[serde(rename = "lastPrice")]
    last_price: String,
    #[serde(rename = "price24hPcnt")]
    price_24h_pct_chg: String,
    #[serde(rename = "highPrice24h")]
    high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    low_price_24h: String,
    #[serde(rename = "volume24h")]
    volume_24h: String,
    #[serde(rename = "turnover24h")]
    turnover_24h: String,
}

/// Ticker information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickerInfo {
    /// Symbol
    pub symbol: String,
    /// Last traded price
    pub last_price: f64,
    /// 24h price change percentage
    pub price_24h_pct: f64,
    /// 24h high
    pub high_24h: f64,
    /// 24h low
    pub low_24h: f64,
    /// 24h volume
    pub volume_24h: f64,
    /// 24h turnover in quote currency
    pub turnover_24h: f64,
}

/// Default trading pairs for crypto analysis.
pub fn default_symbols() -> Vec<&'static str> {
    vec![
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "DOGEUSDT",
        "AVAXUSDT",
        "DOTUSDT",
        "LINKUSDT",
        "MATICUSDT",
        "UNIUSDT",
        "ATOMUSDT",
        "LTCUSDT",
        "ETCUSDT",
    ]
}

/// DeFi tokens for sector-based analysis.
pub fn defi_symbols() -> Vec<&'static str> {
    vec![
        "UNIUSDT",
        "AAVEUSDT",
        "LINKUSDT",
        "MKRUSDT",
        "COMPUSDT",
        "SNXUSDT",
        "CRVUSDT",
        "SUSHIUSDT",
        "YFIUSDT",
        "1INCHUSDT",
    ]
}

/// Layer 1 blockchain tokens.
pub fn layer1_symbols() -> Vec<&'static str> {
    vec![
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "ADAUSDT",
        "AVAXUSDT",
        "DOTUSDT",
        "ATOMUSDT",
        "NEARUSDT",
        "ALGOUSDT",
        "APTUSDT",
    ]
}

/// Layer 2 / scaling tokens.
pub fn layer2_symbols() -> Vec<&'static str> {
    vec![
        "MATICUSDT",
        "ARBUSDT",
        "OPUSDT",
        "IMXUSDT",
        "MANTAUSDT",
    ]
}

/// Convert interval string to minutes.
pub fn interval_to_minutes(interval: &str) -> Option<u64> {
    match interval {
        "1" => Some(1),
        "3" => Some(3),
        "5" => Some(5),
        "15" => Some(15),
        "30" => Some(30),
        "60" => Some(60),
        "120" => Some(120),
        "240" => Some(240),
        "360" => Some(360),
        "720" => Some(720),
        "D" => Some(1440),
        "W" => Some(10080),
        "M" => Some(43200),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_to_minutes() {
        assert_eq!(interval_to_minutes("1"), Some(1));
        assert_eq!(interval_to_minutes("60"), Some(60));
        assert_eq!(interval_to_minutes("D"), Some(1440));
        assert_eq!(interval_to_minutes("invalid"), None);
    }

    #[test]
    fn test_default_symbols() {
        let symbols = default_symbols();
        assert!(symbols.contains(&"BTCUSDT"));
        assert!(symbols.contains(&"ETHUSDT"));
    }

    // Integration tests would require network access
    // #[tokio::test]
    // async fn test_fetch_klines() {
    //     let client = BybitClient::new();
    //     let candles = client.fetch_klines("BTCUSDT", "60", 10).await.unwrap();
    //     assert!(!candles.is_empty());
    // }
}
