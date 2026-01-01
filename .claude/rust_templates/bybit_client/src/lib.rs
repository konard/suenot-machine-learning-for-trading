//! Bybit API Client for Cryptocurrency Market Data
//!
//! This module provides a reusable client for fetching historical and real-time
//! cryptocurrency data from Bybit exchange.
//!
//! # Examples
//!
//! ```no_run
//! use bybit_client::{BybitClient, Interval};
//! use chrono::{Utc, Duration};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = BybitClient::new();
//!
//!     // Fetch 1-hour candles for BTC/USDT
//!     let end_time = Utc::now();
//!     let start_time = end_time - Duration::days(7);
//!
//!     let candles = client
//!         .get_klines("BTCUSDT", Interval::OneHour, start_time, end_time)
//!         .await?;
//!
//!     println!("Fetched {} candles", candles.len());
//!     Ok(())
//! }
//! ```

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration as StdDuration;
use tokio::time::sleep;

/// Bybit API base URL for V5 API
const BYBIT_API_V5: &str = "https://api.bybit.com/v5";

/// Rate limit: Max requests per minute
const RATE_LIMIT_PER_MINUTE: u32 = 120;

/// Time interval for candlestick data
#[derive(Debug, Clone, Copy)]
pub enum Interval {
    OneMinute,
    ThreeMinutes,
    FiveMinutes,
    FifteenMinutes,
    ThirtyMinutes,
    OneHour,
    TwoHours,
    FourHours,
    SixHours,
    TwelveHours,
    OneDay,
    OneWeek,
    OneMonth,
}

impl Interval {
    /// Convert interval to Bybit API format
    pub fn as_str(&self) -> &'static str {
        match self {
            Interval::OneMinute => "1",
            Interval::ThreeMinutes => "3",
            Interval::FiveMinutes => "5",
            Interval::FifteenMinutes => "15",
            Interval::ThirtyMinutes => "30",
            Interval::OneHour => "60",
            Interval::TwoHours => "120",
            Interval::FourHours => "240",
            Interval::SixHours => "360",
            Interval::TwelveHours => "720",
            Interval::OneDay => "D",
            Interval::OneWeek => "W",
            Interval::OneMonth => "M",
        }
    }
}

/// Candlestick (OHLCV) data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Opening timestamp (milliseconds)
    pub timestamp: i64,
    /// Opening price
    pub open: f64,
    /// Highest price
    pub high: f64,
    /// Lowest price
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Trading volume
    pub volume: f64,
    /// Turnover (volume * price)
    pub turnover: f64,
}

/// Response from Bybit klines endpoint
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
    category: String,
    symbol: String,
    list: Vec<Vec<String>>,
}

/// Market ticker data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    pub symbol: String,
    pub last_price: f64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub volume_24h: f64,
    pub price_change_percent_24h: f64,
}

/// Bybit API client
pub struct BybitClient {
    client: Client,
    rate_limiter: RateLimiter,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(StdDuration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            rate_limiter: RateLimiter::new(RATE_LIMIT_PER_MINUTE),
        }
    }

    /// Fetch historical candlestick data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Time interval for candles
    /// * `start_time` - Start of time range
    /// * `end_time` - End of time range
    ///
    /// # Returns
    /// Vector of candles sorted by timestamp (oldest first)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: Interval,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<Candle>> {
        let mut all_candles = Vec::new();
        let mut current_start = start_time;

        // Bybit returns max 1000 candles per request
        let max_candles_per_request = 1000;

        while current_start < end_time {
            // Rate limiting
            self.rate_limiter.wait_if_needed().await;

            let url = format!(
                "{}/market/kline?category=spot&symbol={}&interval={}&start={}&end={}&limit={}",
                BYBIT_API_V5,
                symbol,
                interval.as_str(),
                current_start.timestamp_millis(),
                end_time.timestamp_millis(),
                max_candles_per_request
            );

            let response = self
                .client
                .get(&url)
                .send()
                .await
                .context("Failed to send request to Bybit")?;

            let kline_response: KlineResponse = response
                .json()
                .await
                .context("Failed to parse Bybit response")?;

            if kline_response.ret_code != 0 {
                anyhow::bail!(
                    "Bybit API error: {} (code: {})",
                    kline_response.ret_msg,
                    kline_response.ret_code
                );
            }

            if kline_response.result.list.is_empty() {
                break;
            }

            // Parse candles
            for kline in &kline_response.result.list {
                if kline.len() < 7 {
                    continue;
                }

                let candle = Candle {
                    timestamp: kline[0].parse()?,
                    open: kline[1].parse()?,
                    high: kline[2].parse()?,
                    low: kline[3].parse()?,
                    close: kline[4].parse()?,
                    volume: kline[5].parse()?,
                    turnover: kline[6].parse()?,
                };

                all_candles.push(candle);
            }

            // Move to next batch
            if let Some(last_candle) = all_candles.last() {
                current_start = DateTime::from_timestamp_millis(last_candle.timestamp)
                    .context("Invalid timestamp")?;
                current_start = current_start + chrono::Duration::milliseconds(1);
            } else {
                break;
            }

            // Check if we've reached the end
            if kline_response.result.list.len() < max_candles_per_request {
                break;
            }
        }

        // Sort by timestamp (oldest first)
        all_candles.sort_by_key(|c| c.timestamp);

        Ok(all_candles)
    }

    /// Get current market ticker
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        self.rate_limiter.wait_if_needed().await;

        let url = format!(
            "{}/market/tickers?category=spot&symbol={}",
            BYBIT_API_V5, symbol
        );

        let response = self.client.get(&url).send().await?;
        let json: serde_json::Value = response.json().await?;

        let ret_code = json["retCode"].as_i64().unwrap_or(-1);
        if ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", json["retMsg"]);
        }

        let ticker_data = &json["result"]["list"][0];

        Ok(Ticker {
            symbol: ticker_data["symbol"]
                .as_str()
                .unwrap_or(symbol)
                .to_string(),
            last_price: ticker_data["lastPrice"]
                .as_str()
                .unwrap_or("0")
                .parse()?,
            bid_price: ticker_data["bid1Price"]
                .as_str()
                .unwrap_or("0")
                .parse()?,
            ask_price: ticker_data["ask1Price"]
                .as_str()
                .unwrap_or("0")
                .parse()?,
            volume_24h: ticker_data["volume24h"]
                .as_str()
                .unwrap_or("0")
                .parse()?,
            price_change_percent_24h: ticker_data["price24hPcnt"]
                .as_str()
                .unwrap_or("0")
                .parse()?,
        })
    }

    /// Get list of available symbols
    pub async fn get_symbols(&self) -> Result<Vec<String>> {
        self.rate_limiter.wait_if_needed().await;

        let url = format!("{}/market/instruments-info?category=spot", BYBIT_API_V5);

        let response = self.client.get(&url).send().await?;
        let json: serde_json::Value = response.json().await?;

        let ret_code = json["retCode"].as_i64().unwrap_or(-1);
        if ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", json["retMsg"]);
        }

        let symbols: Vec<String> = json["result"]["list"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|item| item["symbol"].as_str().map(|s| s.to_string()))
            .collect();

        Ok(symbols)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple rate limiter
struct RateLimiter {
    requests_per_minute: u32,
    last_request: std::sync::Mutex<Option<std::time::Instant>>,
}

impl RateLimiter {
    fn new(requests_per_minute: u32) -> Self {
        Self {
            requests_per_minute,
            last_request: std::sync::Mutex::new(None),
        }
    }

    async fn wait_if_needed(&self) {
        let min_interval = StdDuration::from_secs(60) / self.requests_per_minute;

        let mut last = self.last_request.lock().unwrap();
        if let Some(last_time) = *last {
            let elapsed = last_time.elapsed();
            if elapsed < min_interval {
                let wait_time = min_interval - elapsed;
                drop(last); // Release lock before sleeping
                sleep(wait_time).await;
                let mut last = self.last_request.lock().unwrap();
                *last = Some(std::time::Instant::now());
            } else {
                *last = Some(std::time::Instant::now());
            }
        } else {
            *last = Some(std::time::Instant::now());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = BybitClient::new();
        assert!(true); // Client created successfully
    }

    #[tokio::test]
    #[ignore] // Ignore by default as it requires network
    async fn test_get_symbols() {
        let client = BybitClient::new();
        let symbols = client.get_symbols().await.unwrap();
        assert!(!symbols.is_empty());
        assert!(symbols.contains(&"BTCUSDT".to_string()));
    }

    #[tokio::test]
    #[ignore] // Ignore by default as it requires network
    async fn test_get_ticker() {
        let client = BybitClient::new();
        let ticker = client.get_ticker("BTCUSDT").await.unwrap();
        assert_eq!(ticker.symbol, "BTCUSDT");
        assert!(ticker.last_price > 0.0);
    }

    #[tokio::test]
    #[ignore] // Ignore by default as it requires network
    async fn test_get_klines() {
        let client = BybitClient::new();
        let end_time = Utc::now();
        let start_time = end_time - chrono::Duration::hours(24);

        let candles = client
            .get_klines("BTCUSDT", Interval::OneHour, start_time, end_time)
            .await
            .unwrap();

        assert!(!candles.is_empty());
        assert!(candles.len() <= 24);

        // Check candles are sorted
        for i in 1..candles.len() {
            assert!(candles[i].timestamp > candles[i - 1].timestamp);
        }
    }
}
