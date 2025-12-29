//! Bybit API client for fetching market data
//!
//! This module provides async methods to fetch OHLCV (kline) data
//! from the Bybit cryptocurrency exchange API v5.

use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::Deserialize;
use tracing::{info, warn};

use super::ohlcv::{OHLCVData, OHLCVDataset};

/// Bybit API base URL
const BYBIT_API_BASE: &str = "https://api.bybit.com";

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: T,
}

/// Kline result structure
#[derive(Debug, Deserialize)]
struct KlineResult {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

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
        Self {
            client: Client::new(),
            base_url: BYBIT_API_BASE.to_string(),
        }
    }

    /// Create a new Bybit client with custom base URL (for testnet)
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Create client for Bybit testnet
    pub fn testnet() -> Self {
        Self::with_base_url("https://api-testnet.bybit.com")
    }

    /// Fetch kline (OHLCV) data from Bybit
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Kline interval: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
    /// * `start_time` - Start timestamp in milliseconds (optional)
    /// * `end_time` - End timestamp in milliseconds (optional)
    /// * `limit` - Number of records to fetch (max 1000, default 200)
    ///
    /// # Returns
    ///
    /// Vector of OHLCV data points
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        start_time: Option<i64>,
        end_time: Option<i64>,
        limit: Option<u32>,
    ) -> Result<Vec<OHLCVData>> {
        let mut params = vec![
            ("category", "linear".to_string()),
            ("symbol", symbol.to_string()),
            ("interval", interval.to_string()),
            ("limit", limit.unwrap_or(200).min(1000).to_string()),
        ];

        if let Some(start) = start_time {
            params.push(("start", start.to_string()));
        }
        if let Some(end) = end_time {
            params.push(("end", end.to_string()));
        }

        let url = format!("{}/v5/market/kline", self.base_url);

        info!(
            "Fetching klines for {} interval {} from Bybit",
            symbol, interval
        );

        let response = self
            .client
            .get(&url)
            .query(&params)
            .send()
            .await?
            .json::<BybitResponse<KlineResult>>()
            .await?;

        if response.ret_code != 0 {
            return Err(anyhow!(
                "Bybit API error: {} (code: {})",
                response.ret_msg,
                response.ret_code
            ));
        }

        let data: Vec<OHLCVData> = response
            .result
            .list
            .iter()
            .filter_map(|kline| {
                if kline.len() >= 7 {
                    Some(OHLCVData::new(
                        kline[0].parse().ok()?,
                        kline[1].parse().ok()?,
                        kline[2].parse().ok()?,
                        kline[3].parse().ok()?,
                        kline[4].parse().ok()?,
                        kline[5].parse().ok()?,
                        kline[6].parse().ok()?,
                    ))
                } else {
                    warn!("Invalid kline data: {:?}", kline);
                    None
                }
            })
            .collect();

        // Bybit returns data in descending order, reverse to ascending
        let mut data = data;
        data.reverse();

        info!("Fetched {} klines for {}", data.len(), symbol);
        Ok(data)
    }

    /// Fetch historical klines with pagination
    ///
    /// This method handles pagination to fetch more than 1000 records
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
    /// Complete OHLCV dataset
    pub async fn fetch_historical_klines(
        &self,
        symbol: &str,
        interval: &str,
        start_time: i64,
        end_time: i64,
    ) -> Result<OHLCVDataset> {
        let mut all_data: Vec<OHLCVData> = Vec::new();
        let mut current_end = end_time;
        let limit = 1000u32;

        // Calculate interval in milliseconds for pagination
        let interval_ms = Self::interval_to_ms(interval)?;

        loop {
            let data = self
                .fetch_klines(symbol, interval, Some(start_time), Some(current_end), Some(limit))
                .await?;

            if data.is_empty() {
                break;
            }

            let oldest_timestamp = data.first().map(|d| d.timestamp).unwrap_or(start_time);

            // Prepend new data (older) to existing data
            let mut new_data = data;
            new_data.extend(all_data);
            all_data = new_data;

            // Check if we've reached the start
            if oldest_timestamp <= start_time {
                break;
            }

            // Move the end time to fetch older data
            current_end = oldest_timestamp - interval_ms;

            // Add a small delay to avoid rate limiting
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        // Remove duplicates and sort by timestamp
        all_data.sort_by_key(|d| d.timestamp);
        all_data.dedup_by_key(|d| d.timestamp);

        // Filter to requested range
        all_data.retain(|d| d.timestamp >= start_time && d.timestamp <= end_time);

        info!(
            "Fetched {} total klines for {} from {} to {}",
            all_data.len(),
            symbol,
            start_time,
            end_time
        );

        Ok(OHLCVDataset::new(
            symbol.to_string(),
            interval.to_string(),
            all_data,
        ))
    }

    /// Convert interval string to milliseconds
    fn interval_to_ms(interval: &str) -> Result<i64> {
        let ms = match interval {
            "1" => 60_000,
            "3" => 3 * 60_000,
            "5" => 5 * 60_000,
            "15" => 15 * 60_000,
            "30" => 30 * 60_000,
            "60" => 60 * 60_000,
            "120" => 120 * 60_000,
            "240" => 240 * 60_000,
            "360" => 360 * 60_000,
            "720" => 720 * 60_000,
            "D" => 24 * 60 * 60_000,
            "W" => 7 * 24 * 60 * 60_000,
            "M" => 30 * 24 * 60 * 60_000,
            _ => return Err(anyhow!("Invalid interval: {}", interval)),
        };
        Ok(ms)
    }

    /// Get available trading symbols (spot market)
    pub async fn get_symbols(&self) -> Result<Vec<String>> {
        let url = format!("{}/v5/market/instruments-info", self.base_url);

        let response: serde_json::Value = self
            .client
            .get(&url)
            .query(&[("category", "linear")])
            .send()
            .await?
            .json()
            .await?;

        let symbols: Vec<String> = response["result"]["list"]
            .as_array()
            .ok_or_else(|| anyhow!("Invalid response format"))?
            .iter()
            .filter_map(|item| item["symbol"].as_str().map(String::from))
            .collect();

        Ok(symbols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_to_ms() {
        assert_eq!(BybitClient::interval_to_ms("1").unwrap(), 60_000);
        assert_eq!(BybitClient::interval_to_ms("60").unwrap(), 3_600_000);
        assert_eq!(BybitClient::interval_to_ms("D").unwrap(), 86_400_000);
    }
}
