//! Bybit API client implementation

use super::types::{Interval, KlineResponse, Symbol};
use crate::data::Candle;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use reqwest::Client;
use tracing::{debug, info};

/// Bybit API client for fetching market data
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
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Create client with custom base URL (for testnet)
    pub fn with_testnet() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api-testnet.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Kline interval
    /// * `start` - Start time
    /// * `end` - End time
    /// * `limit` - Number of records (max 1000)
    pub async fn get_klines(
        &self,
        symbol: &Symbol,
        interval: Interval,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
        limit: Option<u32>,
    ) -> Result<Vec<Candle>> {
        let mut url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}",
            self.base_url,
            symbol.as_ref(),
            interval.as_str()
        );

        if let Some(s) = start {
            url.push_str(&format!("&start={}", s.timestamp_millis()));
        }
        if let Some(e) = end {
            url.push_str(&format!("&end={}", e.timestamp_millis()));
        }
        if let Some(l) = limit {
            url.push_str(&format!("&limit={}", l.min(1000)));
        }

        debug!("Fetching klines from: {}", url);

        let response: KlineResponse = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to send request")?
            .json()
            .await
            .context("Failed to parse response")?;

        if response.ret_code != 0 {
            anyhow::bail!("API error: {} - {}", response.ret_code, response.ret_msg);
        }

        // Convert to Candle format and reverse (API returns newest first)
        let mut candles: Vec<Candle> = response
            .result
            .list
            .into_iter()
            .map(|k| Candle {
                timestamp: k.timestamp(),
                open: k.open(),
                high: k.high(),
                low: k.low(),
                close: k.close(),
                volume: k.volume(),
            })
            .collect();

        candles.reverse();
        Ok(candles)
    }

    /// Fetch historical data with pagination
    ///
    /// Automatically handles pagination to fetch more than 1000 candles
    pub async fn get_historical_klines(
        &self,
        symbol: &Symbol,
        interval: Interval,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<Candle>> {
        let mut all_candles = Vec::new();
        let mut current_end = end;
        let interval_ms = interval.duration_ms();

        info!(
            "Fetching {} klines from {} to {}",
            symbol.as_ref(),
            start,
            end
        );

        loop {
            let candles = self
                .get_klines(symbol, interval, Some(start), Some(current_end), Some(1000))
                .await?;

            if candles.is_empty() {
                break;
            }

            let oldest_ts = candles.first().map(|c| c.timestamp).unwrap_or(0);

            // Prepend candles (we're going backwards in time)
            let mut new_candles = candles;
            new_candles.append(&mut all_candles);
            all_candles = new_candles;

            // Check if we've reached the start
            if oldest_ts <= start.timestamp_millis() {
                break;
            }

            // Move end time back
            current_end = DateTime::from_timestamp_millis(oldest_ts - interval_ms)
                .unwrap_or(start);

            // Small delay to avoid rate limiting
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        // Filter to exact time range
        all_candles.retain(|c| {
            c.timestamp >= start.timestamp_millis() && c.timestamp <= end.timestamp_millis()
        });

        info!("Fetched {} candles total", all_candles.len());
        Ok(all_candles)
    }

    /// Get available trading symbols
    pub async fn get_symbols(&self) -> Result<Vec<String>> {
        let url = format!("{}/v5/market/instruments-info?category=spot", self.base_url);

        #[derive(serde::Deserialize)]
        struct Response {
            result: ResultData,
        }

        #[derive(serde::Deserialize)]
        struct ResultData {
            list: Vec<InstrumentInfo>,
        }

        #[derive(serde::Deserialize)]
        struct InstrumentInfo {
            symbol: String,
        }

        let response: Response = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        Ok(response.result.list.into_iter().map(|i| i.symbol).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_klines() {
        let client = BybitClient::new();
        let candles = client
            .get_klines(&Symbol::btcusdt(), Interval::Hour1, None, None, Some(10))
            .await
            .unwrap();

        assert!(!candles.is_empty());
        assert!(candles.len() <= 10);
    }
}
