//! Bybit API client for fetching cryptocurrency data.

use super::types::{Candle, PriceSeries};
use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use serde::Deserialize;
use std::collections::HashMap;

/// Base URL for Bybit API v5.
const BYBIT_API_BASE: &str = "https://api.bybit.com";

/// Popular cryptocurrency pairs on Bybit.
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
    "LTCUSDT",
    "UNIUSDT",
    "NEARUSDT",
    "APTUSDT",
    "ARBUSDT",
];

/// Bybit API client.
#[derive(Debug, Clone)]
pub struct BybitClient {
    client: reqwest::Client,
    base_url: String,
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

/// Kline result from Bybit API.
#[derive(Debug, Deserialize)]
struct KlineResult {
    #[allow(dead_code)]
    symbol: String,
    #[allow(dead_code)]
    category: String,
    list: Vec<Vec<String>>,
}

impl BybitClient {
    /// Create a new Bybit client.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: BYBIT_API_BASE.to_string(),
        }
    }

    /// Create client with custom base URL (for testing).
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Fetch kline (candlestick) data.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Time interval ("1", "5", "15", "60", "240", "D", "W")
    /// * `start` - Optional start time
    /// * `end` - Optional end time
    /// * `limit` - Maximum number of candles (default 200, max 1000)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
        limit: Option<u32>,
    ) -> Result<PriceSeries> {
        let mut params = HashMap::new();
        params.insert("category", "spot".to_string());
        params.insert("symbol", symbol.to_string());
        params.insert("interval", interval.to_string());

        if let Some(start_time) = start {
            params.insert("start", start_time.timestamp_millis().to_string());
        }

        if let Some(end_time) = end {
            params.insert("end", end_time.timestamp_millis().to_string());
        }

        params.insert("limit", limit.unwrap_or(200).to_string());

        let url = format!("{}/v5/market/kline", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&params)
            .send()
            .await
            .context("Failed to send request to Bybit API")?;

        let response_text = response.text().await?;

        let bybit_response: BybitResponse<KlineResult> =
            serde_json::from_str(&response_text).context("Failed to parse Bybit response")?;

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

    /// Fetch data for multiple symbols.
    pub async fn fetch_multi_asset(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: u32,
    ) -> Result<HashMap<String, PriceSeries>> {
        let mut result = HashMap::new();

        for symbol in symbols {
            match self.get_klines(symbol, interval, None, None, Some(limit)).await {
                Ok(series) => {
                    result.insert(symbol.to_string(), series);
                }
                Err(e) => {
                    log::warn!("Failed to fetch {}: {}", symbol, e);
                }
            }

            // Rate limit protection
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        Ok(result)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Get recommended universe for C-Mamba strategies.
pub fn get_cmamba_universe() -> Vec<&'static str> {
    CRYPTO_UNIVERSE[..10].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crypto_universe() {
        let universe = get_cmamba_universe();
        assert_eq!(universe.len(), 10);
        assert!(universe.contains(&"BTCUSDT"));
        assert!(universe.contains(&"ETHUSDT"));
    }
}
