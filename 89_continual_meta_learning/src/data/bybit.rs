//! Bybit API client for fetching cryptocurrency data.
//!
//! This module provides an async client for fetching OHLCV data from Bybit.

use reqwest::Client;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use chrono::{DateTime, Utc};

/// Errors that can occur when interacting with Bybit API.
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("API error: {code} - {message}")]
    ApiError { code: i32, message: String },

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Invalid response format")]
    InvalidResponse,
}

/// A single candlestick (OHLCV) data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Start time of the candle (milliseconds).
    pub start_time: i64,
    /// Open price.
    pub open: f64,
    /// High price.
    pub high: f64,
    /// Low price.
    pub low: f64,
    /// Close price.
    pub close: f64,
    /// Trading volume.
    pub volume: f64,
    /// Turnover (quote volume).
    pub turnover: f64,
}

impl Kline {
    /// Get the timestamp as DateTime.
    pub fn datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(self.start_time)
            .unwrap_or_else(|| Utc::now())
    }

    /// Calculate typical price (HLC/3).
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate price range.
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate body size (absolute difference between open and close).
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Check if candle is bullish.
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Calculate return from open to close.
    pub fn returns(&self) -> f64 {
        if self.open > 0.0 {
            (self.close - self.open) / self.open
        } else {
            0.0
        }
    }
}

/// Bybit API response structure.
#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: Option<KlineResult>,
}

#[derive(Debug, Deserialize)]
struct KlineResult {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

/// Time interval for candlestick data.
#[derive(Debug, Clone, Copy)]
pub enum Interval {
    /// 1 minute.
    Min1,
    /// 3 minutes.
    Min3,
    /// 5 minutes.
    Min5,
    /// 15 minutes.
    Min15,
    /// 30 minutes.
    Min30,
    /// 1 hour.
    Hour1,
    /// 2 hours.
    Hour2,
    /// 4 hours.
    Hour4,
    /// 6 hours.
    Hour6,
    /// 12 hours.
    Hour12,
    /// 1 day.
    Day1,
    /// 1 week.
    Week1,
    /// 1 month.
    Month1,
}

impl Interval {
    /// Convert to Bybit API string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Interval::Min1 => "1",
            Interval::Min3 => "3",
            Interval::Min5 => "5",
            Interval::Min15 => "15",
            Interval::Min30 => "30",
            Interval::Hour1 => "60",
            Interval::Hour2 => "120",
            Interval::Hour4 => "240",
            Interval::Hour6 => "360",
            Interval::Hour12 => "720",
            Interval::Day1 => "D",
            Interval::Week1 => "W",
            Interval::Month1 => "M",
        }
    }

    /// Get interval duration in milliseconds.
    pub fn duration_ms(&self) -> i64 {
        match self {
            Interval::Min1 => 60_000,
            Interval::Min3 => 180_000,
            Interval::Min5 => 300_000,
            Interval::Min15 => 900_000,
            Interval::Min30 => 1_800_000,
            Interval::Hour1 => 3_600_000,
            Interval::Hour2 => 7_200_000,
            Interval::Hour4 => 14_400_000,
            Interval::Hour6 => 21_600_000,
            Interval::Hour12 => 43_200_000,
            Interval::Day1 => 86_400_000,
            Interval::Week1 => 604_800_000,
            Interval::Month1 => 2_592_000_000,
        }
    }
}

/// Category of trading product.
#[derive(Debug, Clone, Copy, Default)]
pub enum Category {
    /// Spot trading.
    #[default]
    Spot,
    /// Linear perpetual (USDT-margined).
    Linear,
    /// Inverse perpetual (coin-margined).
    Inverse,
}

impl Category {
    /// Convert to Bybit API string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Category::Spot => "spot",
            Category::Linear => "linear",
            Category::Inverse => "inverse",
        }
    }
}

/// Bybit API client.
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

    /// Create a client with custom base URL (for testnet).
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Create a testnet client.
    pub fn testnet() -> Self {
        Self::with_base_url("https://api-testnet.bybit.com")
    }

    /// Fetch kline (candlestick) data.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Time interval
    /// * `category` - Product category
    /// * `limit` - Maximum number of candles (max 1000)
    /// * `start` - Start timestamp in milliseconds (optional)
    /// * `end` - End timestamp in milliseconds (optional)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: Interval,
        category: Category,
        limit: Option<u32>,
        start: Option<i64>,
        end: Option<i64>,
    ) -> Result<Vec<Kline>, BybitError> {
        let mut url = format!(
            "{}/v5/market/kline?category={}&symbol={}&interval={}",
            self.base_url,
            category.as_str(),
            symbol,
            interval.as_str()
        );

        if let Some(l) = limit {
            url.push_str(&format!("&limit={}", l.min(1000)));
        }
        if let Some(s) = start {
            url.push_str(&format!("&start={}", s));
        }
        if let Some(e) = end {
            url.push_str(&format!("&end={}", e));
        }

        let response: BybitResponse = self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let result = response.result.ok_or(BybitError::InvalidResponse)?;

        let klines: Vec<Kline> = result
            .list
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 7 {
                    Some(Kline {
                        start_time: row[0].parse().ok()?,
                        open: row[1].parse().ok()?,
                        high: row[2].parse().ok()?,
                        low: row[3].parse().ok()?,
                        close: row[4].parse().ok()?,
                        volume: row[5].parse().ok()?,
                        turnover: row[6].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first, reverse to get chronological order
        let mut klines = klines;
        klines.reverse();

        Ok(klines)
    }

    /// Fetch historical klines with pagination.
    ///
    /// This method handles the 1000 limit by making multiple requests.
    pub async fn get_historical_klines(
        &self,
        symbol: &str,
        interval: Interval,
        category: Category,
        start: i64,
        end: i64,
    ) -> Result<Vec<Kline>, BybitError> {
        let mut all_klines = Vec::new();
        let mut current_end = end;
        let interval_ms = interval.duration_ms();

        while current_end > start {
            let batch = self
                .get_klines(symbol, interval, category, Some(1000), Some(start), Some(current_end))
                .await?;

            if batch.is_empty() {
                break;
            }

            let earliest = batch.first().map(|k| k.start_time).unwrap_or(start);

            // Prepend batch to result (we're going backwards in time)
            all_klines.splice(0..0, batch);

            if earliest <= start {
                break;
            }

            // Move end time back
            current_end = earliest - interval_ms;
        }

        // Filter to exact range
        all_klines.retain(|k| k.start_time >= start && k.start_time <= end);

        Ok(all_klines)
    }

    /// Get available trading symbols.
    pub async fn get_symbols(&self, category: Category) -> Result<Vec<String>, BybitError> {
        let url = format!(
            "{}/v5/market/instruments-info?category={}",
            self.base_url,
            category.as_str()
        );

        let response: serde_json::Value = self.client.get(&url).send().await?.json().await?;

        let symbols = response["result"]["list"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| item["symbol"].as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        Ok(symbols)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for fetching klines with a fluent API.
pub struct KlineBuilder<'a> {
    client: &'a BybitClient,
    symbol: String,
    interval: Interval,
    category: Category,
    limit: Option<u32>,
    start: Option<i64>,
    end: Option<i64>,
}

impl<'a> KlineBuilder<'a> {
    /// Create a new builder.
    pub fn new(client: &'a BybitClient, symbol: &str) -> Self {
        Self {
            client,
            symbol: symbol.to_string(),
            interval: Interval::Hour1,
            category: Category::default(),
            limit: None,
            start: None,
            end: None,
        }
    }

    /// Set the interval.
    pub fn interval(mut self, interval: Interval) -> Self {
        self.interval = interval;
        self
    }

    /// Set the category.
    pub fn category(mut self, category: Category) -> Self {
        self.category = category;
        self
    }

    /// Set the limit.
    pub fn limit(mut self, limit: u32) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set the start time.
    pub fn start(mut self, start: i64) -> Self {
        self.start = Some(start);
        self
    }

    /// Set the end time.
    pub fn end(mut self, end: i64) -> Self {
        self.end = Some(end);
        self
    }

    /// Set the time range.
    pub fn range(mut self, start: i64, end: i64) -> Self {
        self.start = Some(start);
        self.end = Some(end);
        self
    }

    /// Fetch the klines.
    pub async fn fetch(self) -> Result<Vec<Kline>, BybitError> {
        self.client
            .get_klines(
                &self.symbol,
                self.interval,
                self.category,
                self.limit,
                self.start,
                self.end,
            )
            .await
    }

    /// Fetch historical klines with pagination.
    pub async fn fetch_all(self) -> Result<Vec<Kline>, BybitError> {
        let start = self.start.unwrap_or(0);
        let end = self.end.unwrap_or_else(|| Utc::now().timestamp_millis());

        self.client
            .get_historical_klines(&self.symbol, self.interval, self.category, start, end)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_calculations() {
        let kline = Kline {
            start_time: 1700000000000,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        assert!((kline.typical_price() - 103.333).abs() < 0.01);
        assert_eq!(kline.range(), 15.0);
        assert_eq!(kline.body_size(), 5.0);
        assert!(kline.is_bullish());
        assert!((kline.returns() - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_interval() {
        assert_eq!(Interval::Hour1.as_str(), "60");
        assert_eq!(Interval::Day1.as_str(), "D");
        assert_eq!(Interval::Hour1.duration_ms(), 3_600_000);
    }

    #[test]
    fn test_category() {
        assert_eq!(Category::Spot.as_str(), "spot");
        assert_eq!(Category::Linear.as_str(), "linear");
    }
}
