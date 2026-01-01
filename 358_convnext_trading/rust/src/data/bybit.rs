//! Bybit API client for fetching market data

use anyhow::{anyhow, Result};
use chrono::{DateTime, TimeZone, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Bybit API base URL
const BYBIT_API_URL: &str = "https://api.bybit.com";

/// Candlestick interval
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Interval {
    M1,  // 1 minute
    M5,  // 5 minutes
    M15, // 15 minutes
    H1,  // 1 hour
    H4,  // 4 hours
    D1,  // 1 day
}

impl Interval {
    /// Convert to Bybit API string
    pub fn as_str(&self) -> &'static str {
        match self {
            Interval::M1 => "1",
            Interval::M5 => "5",
            Interval::M15 => "15",
            Interval::H1 => "60",
            Interval::H4 => "240",
            Interval::D1 => "D",
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "1m" | "1" => Ok(Interval::M1),
            "5m" | "5" => Ok(Interval::M5),
            "15m" | "15" => Ok(Interval::M15),
            "1h" | "60" => Ok(Interval::H1),
            "4h" | "240" => Ok(Interval::H4),
            "1d" | "d" => Ok(Interval::D1),
            _ => Err(anyhow!("Invalid interval: {}", s)),
        }
    }

    /// Get interval duration in milliseconds
    pub fn duration_ms(&self) -> i64 {
        match self {
            Interval::M1 => 60_000,
            Interval::M5 => 300_000,
            Interval::M15 => 900_000,
            Interval::H1 => 3_600_000,
            Interval::H4 => 14_400_000,
            Interval::D1 => 86_400_000,
        }
    }
}

/// OHLCV Candlestick data
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candle {
    /// Opening timestamp in milliseconds
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

impl Candle {
    /// Get datetime from timestamp
    pub fn datetime(&self) -> DateTime<Utc> {
        Utc.timestamp_millis_opt(self.timestamp).unwrap()
    }

    /// Calculate typical price (HLC average)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate price range
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Check if bullish (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Calculate body size
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }
}

/// Bybit API response for klines
#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

/// Bybit API client
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

    /// Create with custom base URL
    pub fn with_url(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Fetch klines (candlestick data)
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Candlestick interval
    /// * `start_time` - Start time in milliseconds
    /// * `end_time` - End time in milliseconds
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: Interval,
        start_time: i64,
        end_time: i64,
    ) -> Result<Vec<Candle>> {
        let mut all_candles = Vec::new();
        let mut current_end = end_time;

        // Bybit returns max 1000 candles per request
        let limit = 1000;

        loop {
            let url = format!(
                "{}/v5/market/kline?category=linear&symbol={}&interval={}&start={}&end={}&limit={}",
                self.base_url,
                symbol,
                interval.as_str(),
                start_time,
                current_end,
                limit
            );

            debug!("Fetching: {}", url);

            let response: BybitResponse = self.client.get(&url).send().await?.json().await?;

            if response.ret_code != 0 {
                return Err(anyhow!("Bybit API error: {}", response.ret_msg));
            }

            if response.result.list.is_empty() {
                break;
            }

            let candles: Vec<Candle> = response
                .result
                .list
                .iter()
                .filter_map(|row| {
                    if row.len() < 7 {
                        return None;
                    }
                    Some(Candle {
                        timestamp: row[0].parse().ok()?,
                        open: row[1].parse().ok()?,
                        high: row[2].parse().ok()?,
                        low: row[3].parse().ok()?,
                        close: row[4].parse().ok()?,
                        volume: row[5].parse().ok()?,
                        turnover: row[6].parse().ok()?,
                    })
                })
                .collect();

            if candles.is_empty() {
                break;
            }

            // Find earliest timestamp
            let earliest = candles.iter().map(|c| c.timestamp).min().unwrap();

            all_candles.extend(candles);

            // Check if we've fetched all data
            if earliest <= start_time || all_candles.len() >= 10000 {
                break;
            }

            // Update end time for next request
            current_end = earliest - 1;

            // Rate limiting
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        // Sort by timestamp
        all_candles.sort_by_key(|c| c.timestamp);

        // Remove duplicates
        all_candles.dedup_by_key(|c| c.timestamp);

        // Filter to requested range
        all_candles.retain(|c| c.timestamp >= start_time && c.timestamp <= end_time);

        info!("Fetched {} candles for {}", all_candles.len(), symbol);

        Ok(all_candles)
    }

    /// Get latest candle
    pub async fn get_latest(&self, symbol: &str, interval: Interval) -> Result<Candle> {
        let end_time = Utc::now().timestamp_millis();
        let start_time = end_time - interval.duration_ms() * 2;

        let candles = self.get_klines(symbol, interval, start_time, end_time).await?;

        candles
            .last()
            .cloned()
            .ok_or_else(|| anyhow!("No candle data available"))
    }

    /// Get ticker price
    pub async fn get_ticker_price(&self, symbol: &str) -> Result<f64> {
        let url = format!(
            "{}/v5/market/tickers?category=linear&symbol={}",
            self.base_url, symbol
        );

        let response: serde_json::Value = self.client.get(&url).send().await?.json().await?;

        let price = response["result"]["list"][0]["lastPrice"]
            .as_str()
            .ok_or_else(|| anyhow!("Failed to parse price"))?
            .parse::<f64>()?;

        Ok(price)
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
    fn test_interval_parsing() {
        assert_eq!(Interval::from_str("1h").unwrap(), Interval::H1);
        assert_eq!(Interval::from_str("4h").unwrap(), Interval::H4);
        assert_eq!(Interval::from_str("1d").unwrap(), Interval::D1);
    }

    #[test]
    fn test_interval_duration() {
        assert_eq!(Interval::H1.duration_ms(), 3_600_000);
        assert_eq!(Interval::D1.duration_ms(), 86_400_000);
    }

    #[test]
    fn test_candle_methods() {
        let candle = Candle {
            timestamp: 1704067200000,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        assert!(candle.is_bullish());
        assert_eq!(candle.body(), 5.0);
        assert_eq!(candle.range(), 15.0);
        assert!((candle.typical_price() - 103.333).abs() < 0.01);
    }
}
