//! Bybit API client for fetching cryptocurrency market data
//!
//! This module provides a client for interacting with the Bybit exchange API
//! to fetch historical and real-time market data.

use crate::utils::Candle;
use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Bybit API base URL
const BYBIT_API_URL: &str = "https://api.bybit.com";

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
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: BYBIT_API_URL.to_string(),
        }
    }

    /// Create a new client with custom base URL (for testing)
    pub fn with_base_url(base_url: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: base_url.to_string(),
        }
    }

    /// Fetch klines (candlestick data) from Bybit
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Kline interval ("1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M")
    /// * `limit` - Number of candles to fetch (max 1000)
    ///
    /// # Returns
    ///
    /// Vector of Candle structs
    pub async fn get_klines(&self, symbol: &str, interval: &str, limit: usize) -> Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch klines from Bybit")?;

        let data: BybitKlineResponse = response
            .json()
            .await
            .context("Failed to parse Bybit kline response")?;

        if data.ret_code != 0 {
            anyhow::bail!("Bybit API error: {} - {}", data.ret_code, data.ret_msg);
        }

        let mut candles: Vec<Candle> = data
            .result
            .list
            .into_iter()
            .filter_map(|k| {
                let timestamp = k[0].parse::<i64>().ok()?;
                let open = k[1].parse::<f64>().ok()?;
                let high = k[2].parse::<f64>().ok()?;
                let low = k[3].parse::<f64>().ok()?;
                let close = k[4].parse::<f64>().ok()?;
                let volume = k[5].parse::<f64>().ok()?;

                Some(Candle {
                    timestamp: Utc.timestamp_millis_opt(timestamp).single()?,
                    open,
                    high,
                    low,
                    close,
                    volume,
                })
            })
            .collect();

        // Bybit returns newest first, reverse to get chronological order
        candles.reverse();

        Ok(candles)
    }

    /// Fetch ticker data for a symbol
    pub async fn get_ticker(&self, symbol: &str) -> Result<TickerData> {
        let url = format!(
            "{}/v5/market/tickers?category=linear&symbol={}",
            self.base_url, symbol
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch ticker from Bybit")?;

        let data: BybitTickerResponse = response
            .json()
            .await
            .context("Failed to parse Bybit ticker response")?;

        if data.ret_code != 0 {
            anyhow::bail!("Bybit API error: {} - {}", data.ret_code, data.ret_msg);
        }

        let ticker = data
            .result
            .list
            .into_iter()
            .next()
            .context("No ticker data found")?;

        Ok(TickerData {
            symbol: ticker.symbol,
            last_price: ticker.last_price.parse().unwrap_or(0.0),
            bid_price: ticker.bid1_price.parse().unwrap_or(0.0),
            ask_price: ticker.ask1_price.parse().unwrap_or(0.0),
            volume_24h: ticker.volume_24h.parse().unwrap_or(0.0),
            price_change_24h: ticker.price_24h_pcnt.parse().unwrap_or(0.0),
            high_24h: ticker.high_price_24h.parse().unwrap_or(0.0),
            low_24h: ticker.low_price_24h.parse().unwrap_or(0.0),
        })
    }

    /// Fetch multiple klines for extended history
    pub async fn get_extended_klines(
        &self,
        symbol: &str,
        interval: &str,
        total_candles: usize,
    ) -> Result<Vec<Candle>> {
        let mut all_candles = Vec::with_capacity(total_candles);
        let mut end_time: Option<i64> = None;
        let batch_size = 1000;

        while all_candles.len() < total_candles {
            let remaining = total_candles - all_candles.len();
            let limit = remaining.min(batch_size);

            let url = if let Some(end) = end_time {
                format!(
                    "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}&end={}",
                    self.base_url, symbol, interval, limit, end
                )
            } else {
                format!(
                    "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
                    self.base_url, symbol, interval, limit
                )
            };

            let response = self
                .client
                .get(&url)
                .send()
                .await
                .context("Failed to fetch klines from Bybit")?;

            let data: BybitKlineResponse = response
                .json()
                .await
                .context("Failed to parse Bybit kline response")?;

            if data.ret_code != 0 {
                anyhow::bail!("Bybit API error: {} - {}", data.ret_code, data.ret_msg);
            }

            if data.result.list.is_empty() {
                break;
            }

            let candles: Vec<Candle> = data
                .result
                .list
                .iter()
                .filter_map(|k| {
                    let timestamp = k[0].parse::<i64>().ok()?;
                    let open = k[1].parse::<f64>().ok()?;
                    let high = k[2].parse::<f64>().ok()?;
                    let low = k[3].parse::<f64>().ok()?;
                    let close = k[4].parse::<f64>().ok()?;
                    let volume = k[5].parse::<f64>().ok()?;

                    Some(Candle {
                        timestamp: Utc.timestamp_millis_opt(timestamp).single()?,
                        open,
                        high,
                        low,
                        close,
                        volume,
                    })
                })
                .collect();

            // Get the oldest timestamp for next batch
            if let Some(oldest) = data.result.list.last() {
                if let Ok(ts) = oldest[0].parse::<i64>() {
                    end_time = Some(ts - 1);
                }
            }

            all_candles.extend(candles);

            // Rate limiting
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Reverse to get chronological order and trim to exact size
        all_candles.reverse();
        all_candles.truncate(total_candles);

        Ok(all_candles)
    }
}

/// Bybit kline API response structure
#[derive(Debug, Deserialize)]
struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitKlineResult,
}

#[derive(Debug, Deserialize)]
struct BybitKlineResult {
    list: Vec<Vec<String>>,
}

/// Bybit ticker API response structure
#[derive(Debug, Deserialize)]
struct BybitTickerResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitTickerResult,
}

#[derive(Debug, Deserialize)]
struct BybitTickerResult {
    list: Vec<BybitTicker>,
}

#[derive(Debug, Deserialize)]
struct BybitTicker {
    symbol: String,
    #[serde(rename = "lastPrice")]
    last_price: String,
    #[serde(rename = "bid1Price")]
    bid1_price: String,
    #[serde(rename = "ask1Price")]
    ask1_price: String,
    #[serde(rename = "volume24h")]
    volume_24h: String,
    #[serde(rename = "price24hPcnt")]
    price_24h_pcnt: String,
    #[serde(rename = "highPrice24h")]
    high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    low_price_24h: String,
}

/// Ticker data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickerData {
    pub symbol: String,
    pub last_price: f64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub volume_24h: f64,
    pub price_change_24h: f64,
    pub high_24h: f64,
    pub low_24h: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = BybitClient::new();
        assert!(!client.base_url.is_empty());
    }

    #[test]
    fn test_custom_base_url() {
        let client = BybitClient::with_base_url("https://test.api.com");
        assert_eq!(client.base_url, "https://test.api.com");
    }
}
