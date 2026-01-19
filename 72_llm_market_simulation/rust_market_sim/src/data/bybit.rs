//! Bybit Exchange API Client
//!
//! Fetches cryptocurrency market data from Bybit exchange.

use super::types::{Candle, PriceSeries};
use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use serde::Deserialize;
use std::collections::HashMap;

/// Bybit API base URL
const BYBIT_API_BASE: &str = "https://api.bybit.com";

/// Bybit API client
#[derive(Debug, Clone)]
pub struct BybitClient {
    client: reqwest::Client,
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

/// Kline result
#[derive(Debug, Deserialize)]
struct KlineResult {
    #[allow(dead_code)]
    symbol: String,
    #[allow(dead_code)]
    category: String,
    list: Vec<Vec<String>>,
}

/// Ticker result
#[derive(Debug, Deserialize)]
struct TickerResult {
    #[allow(dead_code)]
    category: String,
    list: Vec<TickerInfo>,
}

/// Ticker information
#[derive(Debug, Clone, Deserialize)]
pub struct TickerInfo {
    /// Symbol
    pub symbol: String,
    /// Last price
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    /// 24h high
    #[serde(rename = "highPrice24h")]
    pub high_price_24h: String,
    /// 24h low
    #[serde(rename = "lowPrice24h")]
    pub low_price_24h: String,
    /// 24h volume
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    /// 24h price change percent
    #[serde(rename = "price24hPcnt")]
    pub price_24h_pcnt: String,
}

/// Order book snapshot
#[derive(Debug, Clone, Deserialize)]
struct OrderBookResult {
    #[allow(dead_code)]
    s: String,  // symbol
    b: Vec<Vec<String>>,  // bids
    a: Vec<Vec<String>>,  // asks
    ts: i64,  // timestamp
}

/// Order book data
#[derive(Debug, Clone)]
pub struct OrderBookSnapshot {
    /// Timestamp
    pub timestamp: i64,
    /// Bids (price, quantity)
    pub bids: Vec<(f64, f64)>,
    /// Asks (price, quantity)
    pub asks: Vec<(f64, f64)>,
}

impl BybitClient {
    /// Create new client
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: BYBIT_API_BASE.to_string(),
        }
    }

    /// Create client for testnet
    pub fn testnet() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "https://api-testnet.bybit.com".to_string(),
        }
    }

    /// Get klines (candlestick data)
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Kline interval ("1", "5", "15", "60", "240", "D", "W")
    /// * `start` - Start time (optional)
    /// * `end` - End time (optional)
    /// * `limit` - Number of candles (max 1000)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
        limit: Option<u32>,
    ) -> Result<PriceSeries> {
        let mut params = HashMap::new();
        params.insert("category", "linear".to_string());
        params.insert("symbol", symbol.to_string());
        params.insert("interval", interval.to_string());
        params.insert("limit", limit.unwrap_or(200).to_string());

        if let Some(start_time) = start {
            params.insert("start", start_time.timestamp_millis().to_string());
        }
        if let Some(end_time) = end {
            params.insert("end", end_time.timestamp_millis().to_string());
        }

        let url = format!("{}/v5/market/kline", self.base_url);

        let response = self.client
            .get(&url)
            .query(&params)
            .send()
            .await
            .context("Failed to send request to Bybit API")?;

        let response_text = response.text().await?;
        let bybit_response: BybitResponse<KlineResult> =
            serde_json::from_str(&response_text)
                .context("Failed to parse Bybit response")?;

        if bybit_response.ret_code != 0 {
            anyhow::bail!(
                "Bybit API error: {} (code: {})",
                bybit_response.ret_msg,
                bybit_response.ret_code
            );
        }

        let mut series = PriceSeries::new(symbol.to_string(), interval.to_string());

        // Bybit returns data in reverse order (newest first)
        for kline in bybit_response.result.list.iter().rev() {
            if kline.len() >= 6 {
                let timestamp_ms: i64 = kline[0].parse().unwrap_or(0);
                let timestamp = Utc.timestamp_millis_opt(timestamp_ms).unwrap();

                let candle = Candle {
                    timestamp,
                    open: kline[1].parse().unwrap_or(0.0),
                    high: kline[2].parse().unwrap_or(0.0),
                    low: kline[3].parse().unwrap_or(0.0),
                    close: kline[4].parse().unwrap_or(0.0),
                    volume: kline[5].parse().unwrap_or(0.0),
                    turnover: kline.get(6).and_then(|v| v.parse().ok()),
                };

                series.push(candle);
            }
        }

        Ok(series)
    }

    /// Get ticker information
    pub async fn get_ticker(&self, symbol: &str) -> Result<TickerInfo> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        let response = self.client
            .get(&url)
            .query(&[("category", "linear"), ("symbol", symbol)])
            .send()
            .await
            .context("Failed to send request")?;

        let bybit_response: BybitResponse<TickerResult> = response
            .json()
            .await
            .context("Failed to parse response")?;

        if bybit_response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", bybit_response.ret_msg);
        }

        bybit_response.result.list
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Symbol {} not found", symbol))
    }

    /// Get order book depth
    pub async fn get_orderbook(&self, symbol: &str, limit: u32) -> Result<OrderBookSnapshot> {
        let url = format!("{}/v5/market/orderbook", self.base_url);

        let response = self.client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await
            .context("Failed to send request")?;

        let bybit_response: BybitResponse<OrderBookResult> = response
            .json()
            .await
            .context("Failed to parse response")?;

        if bybit_response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", bybit_response.ret_msg);
        }

        let bids: Vec<(f64, f64)> = bybit_response.result.b
            .iter()
            .map(|b| (
                b[0].parse().unwrap_or(0.0),
                b[1].parse().unwrap_or(0.0),
            ))
            .collect();

        let asks: Vec<(f64, f64)> = bybit_response.result.a
            .iter()
            .map(|a| (
                a[0].parse().unwrap_or(0.0),
                a[1].parse().unwrap_or(0.0),
            ))
            .collect();

        Ok(OrderBookSnapshot {
            timestamp: bybit_response.result.ts,
            bids,
            asks,
        })
    }

    /// Fetch price history (convenience method)
    pub async fn fetch_price_history(
        &self,
        symbol: &str,
        days: u32,
        interval: &str,
    ) -> Result<Vec<f64>> {
        let end = Utc::now();
        let start = end - chrono::Duration::days(days as i64);

        let series = self.get_klines(symbol, interval, Some(start), Some(end), None).await?;

        Ok(series.closes())
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// List of popular crypto pairs
pub const CRYPTO_UNIVERSE: &[&str] = &[
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "MATICUSDT",
    "LINKUSDT",
    "ATOMUSDT",
    "LTCUSDT",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = BybitClient::new();
        assert_eq!(client.base_url, BYBIT_API_BASE);
    }

    #[test]
    fn test_crypto_universe() {
        assert!(CRYPTO_UNIVERSE.contains(&"BTCUSDT"));
        assert!(CRYPTO_UNIVERSE.contains(&"ETHUSDT"));
    }
}
