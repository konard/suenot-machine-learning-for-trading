//! Bybit API client for fetching market data.
//!
//! This module provides an async client for interacting with Bybit's public API.

use super::{MarketData, OHLCV};
use chrono::{DateTime, TimeZone, Utc};
use reqwest::Client;
use serde::Deserialize;
use thiserror::Error;

/// Bybit API base URL
const BYBIT_API_URL: &str = "https://api.bybit.com";

/// Errors that can occur when interacting with Bybit API
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("API error: {message} (code: {code})")]
    ApiError { code: i32, message: String },

    #[error("Invalid response format: {0}")]
    ParseError(String),

    #[error("Invalid symbol: {0}")]
    InvalidSymbol(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,
}

/// Response structure from Bybit kline endpoint
#[derive(Debug, Deserialize)]
struct KlineResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: KlineResult,
}

#[derive(Debug, Deserialize)]
struct KlineResult {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

/// Bybit API client
#[derive(Debug, Clone)]
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: BYBIT_API_URL.to_string(),
        }
    }

    /// Create a new Bybit client with custom base URL (for testing)
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Fetch klines (candlestick data) for multiple symbols
    ///
    /// # Arguments
    ///
    /// * `symbols` - List of trading pair symbols (e.g., ["BTCUSDT", "ETHUSDT"])
    /// * `interval` - Candle interval: "1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"
    /// * `limit` - Number of candles to fetch (max 1000)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use graph_generation_trading::data::BybitClient;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let client = BybitClient::new();
    ///     let symbols = vec!["BTCUSDT", "ETHUSDT"];
    ///     let data = client.fetch_klines(&symbols, "60", 100).await.unwrap();
    /// }
    /// ```
    pub async fn fetch_klines(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: u32,
    ) -> Result<MarketData, BybitError> {
        let limit = limit.min(1000); // Bybit max is 1000

        let symbol_strings: Vec<String> = symbols.iter().map(|s| s.to_string()).collect();
        let mut market_data = MarketData::new(symbol_strings, interval);

        for symbol in symbols {
            let candles = self.fetch_symbol_klines(symbol, interval, limit).await?;
            market_data.add_candles(symbol, candles);
        }

        Ok(market_data)
    }

    /// Fetch klines for a single symbol
    async fn fetch_symbol_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<OHLCV>, BybitError> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response = self.client.get(&url).send().await?;

        if response.status() == 429 {
            return Err(BybitError::RateLimitExceeded);
        }

        let kline_response: KlineResponse = response.json().await?;

        if kline_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: kline_response.ret_code,
                message: kline_response.ret_msg,
            });
        }

        let mut candles: Vec<OHLCV> = kline_response
            .result
            .list
            .iter()
            .filter_map(|item| self.parse_kline_item(item))
            .collect();

        // Bybit returns newest first, reverse to get chronological order
        candles.reverse();

        Ok(candles)
    }

    /// Parse a single kline item from API response
    fn parse_kline_item(&self, item: &[String]) -> Option<OHLCV> {
        if item.len() < 7 {
            return None;
        }

        let timestamp_ms: i64 = item[0].parse().ok()?;
        let timestamp = Utc.timestamp_millis_opt(timestamp_ms).single()?;

        Some(OHLCV {
            timestamp,
            open: item[1].parse().ok()?,
            high: item[2].parse().ok()?,
            low: item[3].parse().ok()?,
            close: item[4].parse().ok()?,
            volume: item[5].parse().ok()?,
            turnover: item[6].parse().ok()?,
        })
    }

    /// Fetch ticker information for a symbol
    pub async fn fetch_ticker(&self, symbol: &str) -> Result<TickerInfo, BybitError> {
        let url = format!(
            "{}/v5/market/tickers?category=linear&symbol={}",
            self.base_url, symbol
        );

        let response = self.client.get(&url).send().await?;
        let ticker_response: TickerResponse = response.json().await?;

        if ticker_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: ticker_response.ret_code,
                message: ticker_response.ret_msg,
            });
        }

        ticker_response
            .result
            .list
            .into_iter()
            .next()
            .ok_or_else(|| BybitError::InvalidSymbol(symbol.to_string()))
    }

    /// Get list of available trading pairs
    pub async fn get_symbols(&self) -> Result<Vec<String>, BybitError> {
        let url = format!(
            "{}/v5/market/instruments-info?category=linear&limit=500",
            self.base_url
        );

        let response = self.client.get(&url).send().await?;
        let instruments: InstrumentsResponse = response.json().await?;

        if instruments.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: instruments.ret_code,
                message: instruments.ret_msg,
            });
        }

        Ok(instruments
            .result
            .list
            .into_iter()
            .map(|i| i.symbol)
            .collect())
    }
}

/// Ticker information
#[derive(Debug, Clone, Deserialize)]
pub struct TickerInfo {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "indexPrice")]
    pub index_price: String,
    #[serde(rename = "markPrice")]
    pub mark_price: String,
    #[serde(rename = "prevPrice24h")]
    pub prev_price_24h: String,
    #[serde(rename = "price24hPcnt")]
    pub price_24h_pcnt: String,
    #[serde(rename = "highPrice24h")]
    pub high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_price_24h: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    #[serde(rename = "turnover24h")]
    pub turnover_24h: String,
}

#[derive(Debug, Deserialize)]
struct TickerResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: TickerResult,
}

#[derive(Debug, Deserialize)]
struct TickerResult {
    category: String,
    list: Vec<TickerInfo>,
}

#[derive(Debug, Deserialize)]
struct InstrumentsResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: InstrumentsResult,
}

#[derive(Debug, Deserialize)]
struct InstrumentsResult {
    category: String,
    list: Vec<InstrumentInfo>,
}

#[derive(Debug, Deserialize)]
struct InstrumentInfo {
    symbol: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_kline_item() {
        let client = BybitClient::new();
        let item = vec![
            "1704067200000".to_string(), // timestamp
            "42000.00".to_string(),       // open
            "42500.00".to_string(),       // high
            "41800.00".to_string(),       // low
            "42300.00".to_string(),       // close
            "1000.5".to_string(),         // volume
            "42150000.00".to_string(),    // turnover
        ];

        let candle = client.parse_kline_item(&item).unwrap();
        assert_eq!(candle.open, 42000.0);
        assert_eq!(candle.close, 42300.0);
        assert_eq!(candle.volume, 1000.5);
    }
}
