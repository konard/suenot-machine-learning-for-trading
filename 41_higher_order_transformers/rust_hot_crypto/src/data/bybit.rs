//! Bybit API client for fetching market data
//!
//! This module provides a client for the Bybit v5 public API.

use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use serde::Deserialize;

use super::types::{Candle, PriceSeries};

/// Bybit API base URL
const BYBIT_API_URL: &str = "https://api.bybit.com";

/// Bybit API client
#[derive(Debug, Clone)]
pub struct BybitClient {
    client: reqwest::Client,
    base_url: String,
}

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
struct ApiResponse<T> {
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

/// Ticker result from Bybit API
#[derive(Debug, Deserialize)]
struct TickerResult {
    category: String,
    list: Vec<TickerItem>,
}

/// Single ticker item
#[derive(Debug, Deserialize)]
struct TickerItem {
    symbol: String,
    #[serde(rename = "lastPrice")]
    last_price: String,
    #[serde(rename = "volume24h")]
    volume_24h: String,
    #[serde(rename = "turnover24h")]
    turnover_24h: String,
    #[serde(rename = "price24hPcnt")]
    price_24h_pcnt: String,
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
            client: reqwest::Client::new(),
            base_url: BYBIT_API_URL.to_string(),
        }
    }

    /// Create a client with custom base URL (for testing)
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Fetch kline (candlestick) data
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Candle interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, M, W)
    /// * `start_time` - Start time (optional)
    /// * `end_time` - End time (optional)
    /// * `limit` - Number of candles (max 1000)
    ///
    /// # Returns
    ///
    /// A `PriceSeries` containing the candle data
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        start_time: Option<i64>,
        end_time: Option<i64>,
        limit: Option<u32>,
    ) -> Result<PriceSeries> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let mut params = vec![
            ("category", "linear".to_string()),
            ("symbol", symbol.to_string()),
            ("interval", interval.to_string()),
        ];

        if let Some(start) = start_time {
            params.push(("start", start.to_string()));
        }
        if let Some(end) = end_time {
            params.push(("end", end.to_string()));
        }
        if let Some(lim) = limit {
            params.push(("limit", lim.min(1000).to_string()));
        }

        let response = self
            .client
            .get(&url)
            .query(&params)
            .send()
            .await
            .context("Failed to fetch klines from Bybit")?;

        let api_response: ApiResponse<KlineResult> = response
            .json()
            .await
            .context("Failed to parse Bybit kline response")?;

        if api_response.ret_code != 0 {
            anyhow::bail!(
                "Bybit API error: {} (code: {})",
                api_response.ret_msg,
                api_response.ret_code
            );
        }

        let candles = self.parse_klines(&api_response.result.list)?;

        Ok(PriceSeries::new(symbol.to_string(), candles))
    }

    /// Parse kline data from API response
    fn parse_klines(&self, data: &[Vec<String>]) -> Result<Vec<Candle>> {
        let mut candles = Vec::with_capacity(data.len());

        for row in data {
            if row.len() < 7 {
                continue;
            }

            let timestamp_ms: i64 = row[0].parse().context("Invalid timestamp")?;
            let timestamp = Utc
                .timestamp_millis_opt(timestamp_ms)
                .single()
                .context("Invalid timestamp value")?;

            let candle = Candle::new(
                timestamp,
                row[1].parse().context("Invalid open price")?,
                row[2].parse().context("Invalid high price")?,
                row[3].parse().context("Invalid low price")?,
                row[4].parse().context("Invalid close price")?,
                row[5].parse().context("Invalid volume")?,
                row[6].parse().context("Invalid quote volume")?,
            );

            candles.push(candle);
        }

        // Bybit returns data in descending order, reverse it
        candles.reverse();

        Ok(candles)
    }

    /// Get current ticker price for a symbol
    pub async fn get_ticker(&self, symbol: &str) -> Result<(f64, f64)> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        let params = [("category", "linear"), ("symbol", symbol)];

        let response = self
            .client
            .get(&url)
            .query(&params)
            .send()
            .await
            .context("Failed to fetch ticker from Bybit")?;

        let api_response: ApiResponse<TickerResult> = response
            .json()
            .await
            .context("Failed to parse Bybit ticker response")?;

        if api_response.ret_code != 0 {
            anyhow::bail!(
                "Bybit API error: {} (code: {})",
                api_response.ret_msg,
                api_response.ret_code
            );
        }

        let ticker = api_response
            .result
            .list
            .first()
            .context("No ticker data returned")?;

        let price: f64 = ticker.last_price.parse().context("Invalid price")?;
        let volume: f64 = ticker.volume_24h.parse().context("Invalid volume")?;

        Ok((price, volume))
    }

    /// Fetch multiple symbols' klines
    pub async fn get_multi_klines(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: Option<u32>,
    ) -> Result<Vec<PriceSeries>> {
        let mut results = Vec::with_capacity(symbols.len());

        for symbol in symbols {
            match self.get_klines(symbol, interval, None, None, limit).await {
                Ok(series) => results.push(series),
                Err(e) => {
                    log::warn!("Failed to fetch {}: {}", symbol, e);
                }
            }

            // Rate limiting - small delay between requests
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        Ok(results)
    }
}

/// Parse interval string to minutes
pub fn parse_interval_minutes(interval: &str) -> u32 {
    match interval {
        "1" => 1,
        "3" => 3,
        "5" => 5,
        "15" => 15,
        "30" => 30,
        "60" => 60,
        "120" => 120,
        "240" => 240,
        "360" => 360,
        "720" => 720,
        "D" => 1440,
        "W" => 10080,
        _ => 60,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_interval() {
        assert_eq!(parse_interval_minutes("1"), 1);
        assert_eq!(parse_interval_minutes("60"), 60);
        assert_eq!(parse_interval_minutes("D"), 1440);
    }
}
