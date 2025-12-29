//! Bybit API client for fetching market data.

use crate::data::Candle;
use anyhow::{anyhow, Result};
use chrono::{DateTime, TimeZone, Utc};
use hmac::{Hmac, Mac};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

/// Bybit API client
pub struct BybitClient {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    api_secret: Option<String>,
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

/// Kline (candlestick) response data
#[derive(Debug, Deserialize)]
struct KlineResult {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

/// Ticker response data
#[derive(Debug, Deserialize)]
struct TickerResult {
    category: String,
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
}

/// Available trading intervals
#[derive(Debug, Clone, Copy, Serialize)]
pub enum Interval {
    #[serde(rename = "1")]
    Min1,
    #[serde(rename = "3")]
    Min3,
    #[serde(rename = "5")]
    Min5,
    #[serde(rename = "15")]
    Min15,
    #[serde(rename = "30")]
    Min30,
    #[serde(rename = "60")]
    Hour1,
    #[serde(rename = "120")]
    Hour2,
    #[serde(rename = "240")]
    Hour4,
    #[serde(rename = "360")]
    Hour6,
    #[serde(rename = "720")]
    Hour12,
    #[serde(rename = "D")]
    Day1,
    #[serde(rename = "W")]
    Week1,
    #[serde(rename = "M")]
    Month1,
}

impl Interval {
    fn as_str(&self) -> &'static str {
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
}

impl BybitClient {
    /// Create a new Bybit client (public endpoints only)
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api.bybit.com".to_string(),
            api_key: None,
            api_secret: None,
        }
    }

    /// Create a new Bybit client with API credentials
    pub fn with_credentials(api_key: String, api_secret: String) -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api.bybit.com".to_string(),
            api_key: Some(api_key),
            api_secret: Some(api_secret),
        }
    }

    /// Use testnet instead of mainnet
    pub fn use_testnet(mut self) -> Self {
        self.base_url = "https://api-testnet.bybit.com".to_string();
        self
    }

    /// Get current timestamp in milliseconds
    fn timestamp_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    /// Sign a request (for private endpoints)
    fn sign(&self, params: &str) -> Result<String> {
        let secret = self
            .api_secret
            .as_ref()
            .ok_or_else(|| anyhow!("API secret not configured"))?;

        let mut mac =
            HmacSha256::new_from_slice(secret.as_bytes()).map_err(|e| anyhow!("HMAC error: {}", e))?;
        mac.update(params.as_bytes());
        let result = mac.finalize();
        Ok(hex::encode(result.into_bytes()))
    }

    /// Fetch historical klines (candlesticks)
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Candlestick interval
    /// * `limit` - Number of candles to fetch (max 1000)
    /// * `start` - Optional start timestamp
    /// * `end` - Optional end timestamp
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: Interval,
        limit: Option<u32>,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Result<Vec<Candle>> {
        let mut params: HashMap<&str, String> = HashMap::new();
        params.insert("category", "spot".to_string());
        params.insert("symbol", symbol.to_string());
        params.insert("interval", interval.as_str().to_string());

        if let Some(l) = limit {
            params.insert("limit", l.min(1000).to_string());
        }
        if let Some(s) = start {
            params.insert("start", (s.timestamp_millis()).to_string());
        }
        if let Some(e) = end {
            params.insert("end", (e.timestamp_millis()).to_string());
        }

        let url = format!("{}/v5/market/kline", self.base_url);
        let response: BybitResponse<KlineResult> = self
            .client
            .get(&url)
            .query(&params)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(anyhow!(
                "Bybit API error: {} - {}",
                response.ret_code,
                response.ret_msg
            ));
        }

        let candles: Vec<Candle> = response
            .result
            .list
            .into_iter()
            .filter_map(|item| {
                if item.len() >= 7 {
                    let timestamp_ms: i64 = item[0].parse().ok()?;
                    let timestamp = Utc.timestamp_millis_opt(timestamp_ms).single()?;
                    Some(Candle::new(
                        timestamp,
                        symbol.to_string(),
                        item[1].parse().ok()?,
                        item[2].parse().ok()?,
                        item[3].parse().ok()?,
                        item[4].parse().ok()?,
                        item[5].parse().ok()?,
                        item[6].parse().ok()?,
                    ))
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns data in descending order, reverse to ascending
        let mut candles = candles;
        candles.reverse();
        Ok(candles)
    }

    /// Fetch current ticker information
    pub async fn get_ticker(&self, symbol: &str) -> Result<(f64, f64, f64, f64, f64)> {
        let url = format!("{}/v5/market/tickers", self.base_url);
        let params = [("category", "spot"), ("symbol", symbol)];

        let response: BybitResponse<TickerResult> = self
            .client
            .get(&url)
            .query(&params)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(anyhow!(
                "Bybit API error: {} - {}",
                response.ret_code,
                response.ret_msg
            ));
        }

        let ticker = response
            .result
            .list
            .first()
            .ok_or_else(|| anyhow!("No ticker data"))?;

        Ok((
            ticker.last_price.parse()?,
            ticker.high_price_24h.parse()?,
            ticker.low_price_24h.parse()?,
            ticker.volume_24h.parse()?,
            ticker.turnover_24h.parse()?,
        ))
    }

    /// Fetch multiple pages of historical data
    pub async fn get_historical_klines(
        &self,
        symbol: &str,
        interval: Interval,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<Candle>> {
        let mut all_candles = Vec::new();
        let mut current_start = start;

        while current_start < end {
            let candles = self
                .get_klines(symbol, interval, Some(1000), Some(current_start), Some(end))
                .await?;

            if candles.is_empty() {
                break;
            }

            let last_timestamp = candles.last().unwrap().timestamp;
            all_candles.extend(candles);

            // Move start to after the last candle
            current_start = last_timestamp + chrono::Duration::milliseconds(1);

            // Small delay to avoid rate limiting
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        Ok(all_candles)
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

    #[tokio::test]
    async fn test_get_ticker() {
        let client = BybitClient::new();
        let result = client.get_ticker("BTCUSDT").await;
        // This test requires network access
        if let Ok((price, high, low, volume, turnover)) = result {
            assert!(price > 0.0);
            assert!(high >= low);
            assert!(volume > 0.0);
            println!(
                "BTC price: {}, 24h high: {}, 24h low: {}, volume: {}",
                price, high, low, volume
            );
        }
    }
}
