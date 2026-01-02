//! Bybit API client implementation

use super::types::{Candle, KlineResponse, TickerResponse};
use anyhow::{Context, Result};
use log::{debug, info};
use reqwest::Client;
use std::time::Duration;

/// Bybit API base URL
const BYBIT_API_URL: &str = "https://api.bybit.com";

/// Client for interacting with Bybit API
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

    /// Create a client with custom base URL (for testing)
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

    /// Fetch kline (candlestick) data
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Kline interval ("1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M")
    /// * `limit` - Number of candles to fetch (max 1000)
    ///
    /// # Returns
    ///
    /// Vector of candles sorted by timestamp (oldest first)
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>> {
        let limit = limit.min(1000);
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

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

        let mut candles: Vec<Candle> = response
            .result
            .list
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 7 {
                    Some(Candle::new(
                        row[0].parse().unwrap_or(0),
                        row[1].parse().unwrap_or(0.0),
                        row[2].parse().unwrap_or(0.0),
                        row[3].parse().unwrap_or(0.0),
                        row[4].parse().unwrap_or(0.0),
                        row[5].parse().unwrap_or(0.0),
                        row[6].parse().unwrap_or(0.0),
                    ))
                } else {
                    None
                }
            })
            .collect();

        // Sort by timestamp (oldest first)
        candles.sort_by_key(|c| c.timestamp);

        info!("Fetched {} candles for {}", candles.len(), symbol);
        Ok(candles)
    }

    /// Fetch klines with pagination to get more historical data
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol
    /// * `interval` - Kline interval
    /// * `total` - Total number of candles to fetch
    ///
    /// # Returns
    ///
    /// Vector of candles sorted by timestamp
    pub async fn fetch_klines_paginated(
        &self,
        symbol: &str,
        interval: &str,
        total: usize,
    ) -> Result<Vec<Candle>> {
        let mut all_candles = Vec::with_capacity(total);
        let mut end_time: Option<i64> = None;

        while all_candles.len() < total {
            let remaining = total - all_candles.len();
            let batch_size = remaining.min(1000);

            let url = match end_time {
                Some(end) => format!(
                    "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}&end={}",
                    self.base_url, symbol, interval, batch_size, end
                ),
                None => format!(
                    "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
                    self.base_url, symbol, interval, batch_size
                ),
            };

            debug!("Fetching batch from: {}", url);

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

            let batch: Vec<Candle> = response
                .result
                .list
                .into_iter()
                .filter_map(|row| {
                    if row.len() >= 7 {
                        Some(Candle::new(
                            row[0].parse().unwrap_or(0),
                            row[1].parse().unwrap_or(0.0),
                            row[2].parse().unwrap_or(0.0),
                            row[3].parse().unwrap_or(0.0),
                            row[4].parse().unwrap_or(0.0),
                            row[5].parse().unwrap_or(0.0),
                            row[6].parse().unwrap_or(0.0),
                        ))
                    } else {
                        None
                    }
                })
                .collect();

            if batch.is_empty() {
                break;
            }

            // Get the earliest timestamp for the next batch
            end_time = batch.iter().map(|c| c.timestamp).min();

            all_candles.extend(batch);

            // Small delay to avoid rate limiting
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Sort by timestamp and deduplicate
        all_candles.sort_by_key(|c| c.timestamp);
        all_candles.dedup_by_key(|c| c.timestamp);

        info!(
            "Fetched {} candles for {} (requested {})",
            all_candles.len(),
            symbol,
            total
        );

        Ok(all_candles)
    }

    /// Fetch current ticker information
    pub async fn fetch_ticker(&self, symbol: &str) -> Result<f64> {
        let url = format!(
            "{}/v5/market/tickers?category=linear&symbol={}",
            self.base_url, symbol
        );

        let response: TickerResponse = self
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

        let price = response
            .result
            .list
            .first()
            .map(|t| t.last_price.parse::<f64>().unwrap_or(0.0))
            .unwrap_or(0.0);

        Ok(price)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = BybitClient::new();
        assert_eq!(client.base_url, BYBIT_API_URL);
    }
}
