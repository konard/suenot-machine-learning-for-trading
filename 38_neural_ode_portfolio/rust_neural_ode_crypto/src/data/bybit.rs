//! # Bybit API Client
//!
//! Asynchronous client for fetching market data from Bybit exchange.
//!
//! ## Example
//!
//! ```rust,no_run
//! use neural_ode_crypto::data::BybitClient;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let client = BybitClient::new();
//!
//!     // Fetch 1000 hourly candles for BTCUSDT
//!     let candles = client.get_klines("BTCUSDT", "60", 1000).await?;
//!
//!     println!("Fetched {} candles", candles.len());
//!     Ok(())
//! }
//! ```

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use super::candles::{Candle, CandleData};

/// Bybit API base URL
const BYBIT_API_URL: &str = "https://api.bybit.com";

/// Bybit API client for fetching market data
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
    /// Create a new Bybit client with default settings
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: BYBIT_API_URL.to_string(),
        }
    }

    /// Create a new Bybit client with custom base URL (for testnet)
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Fetch klines (candlestick data) from Bybit
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Kline interval: "1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"
    /// * `limit` - Number of candles to fetch (max 1000)
    ///
    /// # Returns
    ///
    /// Vector of [`Candle`] structs ordered by time (oldest first)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>> {
        let limit = limit.min(1000); // Bybit max is 1000

        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        debug!("Fetching klines from: {}", url);

        let response: BybitKlineResponse = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to send request to Bybit")?
            .json()
            .await
            .context("Failed to parse Bybit response")?;

        if response.ret_code != 0 {
            anyhow::bail!(
                "Bybit API error: {} (code: {})",
                response.ret_msg,
                response.ret_code
            );
        }

        let mut candles: Vec<Candle> = response
            .result
            .list
            .into_iter()
            .filter_map(|k| Candle::from_bybit_kline(&k).ok())
            .collect();

        // Bybit returns newest first, we want oldest first
        candles.reverse();

        info!(
            "Fetched {} {} candles for {}",
            candles.len(),
            interval,
            symbol
        );

        Ok(candles)
    }

    /// Fetch klines with pagination for historical data
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol
    /// * `interval` - Kline interval
    /// * `start_time` - Start timestamp in milliseconds
    /// * `end_time` - End timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// Vector of [`Candle`] structs covering the time range
    pub async fn get_klines_range(
        &self,
        symbol: &str,
        interval: &str,
        start_time: i64,
        end_time: i64,
    ) -> Result<Vec<Candle>> {
        let mut all_candles = Vec::new();
        let mut current_end = end_time;

        loop {
            let url = format!(
                "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit=1000&end={}",
                self.base_url, symbol, interval, current_end
            );

            let response: BybitKlineResponse = self
                .client
                .get(&url)
                .send()
                .await?
                .json()
                .await?;

            if response.ret_code != 0 {
                warn!("Bybit API error during pagination: {}", response.ret_msg);
                break;
            }

            let candles: Vec<Candle> = response
                .result
                .list
                .into_iter()
                .filter_map(|k| Candle::from_bybit_kline(&k).ok())
                .collect();

            if candles.is_empty() {
                break;
            }

            let oldest_time = candles.last().map(|c| c.open_time).unwrap_or(0);

            all_candles.extend(candles);

            if oldest_time <= start_time {
                break;
            }

            current_end = oldest_time - 1;

            // Rate limiting
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        // Remove duplicates and sort by time
        all_candles.sort_by_key(|c| c.open_time);
        all_candles.dedup_by_key(|c| c.open_time);

        // Filter to requested range
        all_candles.retain(|c| c.open_time >= start_time && c.open_time <= end_time);

        info!(
            "Fetched {} candles for {} from {} to {}",
            all_candles.len(),
            symbol,
            start_time,
            end_time
        );

        Ok(all_candles)
    }

    /// Fetch ticker data for a symbol
    pub async fn get_ticker(&self, symbol: &str) -> Result<TickerData> {
        let url = format!(
            "{}/v5/market/tickers?category=spot&symbol={}",
            self.base_url, symbol
        );

        let response: BybitTickerResponse = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        response
            .result
            .list
            .into_iter()
            .next()
            .context("No ticker data returned")
    }

    /// Fetch multiple tickers at once
    pub async fn get_tickers(&self, symbols: &[&str]) -> Result<Vec<TickerData>> {
        let mut tickers = Vec::new();

        for symbol in symbols {
            match self.get_ticker(symbol).await {
                Ok(ticker) => tickers.push(ticker),
                Err(e) => warn!("Failed to fetch ticker for {}: {}", symbol, e),
            }
            // Rate limiting
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }

        Ok(tickers)
    }
}

// ============================================================================
// Bybit API Response Structures
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BybitKlineResponse {
    ret_code: i32,
    ret_msg: String,
    result: BybitKlineResult,
}

#[derive(Debug, Deserialize)]
struct BybitKlineResult {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BybitTickerResponse {
    ret_code: i32,
    ret_msg: String,
    result: BybitTickerResult,
}

#[derive(Debug, Deserialize)]
struct BybitTickerResult {
    category: String,
    list: Vec<TickerData>,
}

/// Ticker data from Bybit
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TickerData {
    pub symbol: String,
    pub last_price: String,
    pub high_price_24h: String,
    pub low_price_24h: String,
    pub prev_price_24h: String,
    pub volume_24h: String,
    pub turnover_24h: String,
    pub price_24h_pcnt: String,
}

impl TickerData {
    /// Get last price as f64
    pub fn last_price_f64(&self) -> f64 {
        self.last_price.parse().unwrap_or(0.0)
    }

    /// Get 24h price change percentage as f64
    pub fn price_change_24h(&self) -> f64 {
        self.price_24h_pcnt.parse().unwrap_or(0.0)
    }

    /// Get 24h volume as f64
    pub fn volume_24h_f64(&self) -> f64 {
        self.volume_24h.parse().unwrap_or(0.0)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = BybitClient::new();
        assert_eq!(client.base_url, BYBIT_API_URL);
    }

    #[test]
    fn test_custom_base_url() {
        let client = BybitClient::with_base_url("https://api-testnet.bybit.com");
        assert_eq!(client.base_url, "https://api-testnet.bybit.com");
    }

    #[tokio::test]
    #[ignore] // Run manually to avoid hitting the API
    async fn test_get_klines() {
        let client = BybitClient::new();
        let candles = client.get_klines("BTCUSDT", "60", 10).await.unwrap();
        assert!(!candles.is_empty());
        assert!(candles.len() <= 10);
    }
}
