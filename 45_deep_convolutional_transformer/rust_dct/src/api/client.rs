//! Bybit HTTP client implementation

use super::types::{KlineData, KlineResponse};
use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use reqwest::Client;

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
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Create client with custom base URL
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Fetch klines (candlesticks) from Bybit
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Candle interval ("1", "5", "15", "60", "240", "D", "W")
    /// * `limit` - Maximum number of candles (max 1000)
    /// * `start_time` - Optional start timestamp in milliseconds
    /// * `end_time` - Optional end timestamp in milliseconds
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
        start_time: Option<i64>,
        end_time: Option<i64>,
    ) -> Result<Vec<KlineData>> {
        let mut url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit.min(1000)
        );

        if let Some(start) = start_time {
            url.push_str(&format!("&start={}", start));
        }
        if let Some(end) = end_time {
            url.push_str(&format!("&end={}", end));
        }

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
            anyhow::bail!("API error: {}", response.ret_msg);
        }

        let klines: Vec<KlineData> = response
            .result
            .list
            .iter()
            .filter_map(|item| KlineData::from_list(item))
            .collect();

        // Sort by timestamp ascending
        let mut klines = klines;
        klines.sort_by_key(|k| k.timestamp);

        Ok(klines)
    }

    /// Fetch historical data with pagination
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair
    /// * `interval` - Candle interval
    /// * `days` - Number of days of history
    pub async fn get_historical_data(
        &self,
        symbol: &str,
        interval: &str,
        days: u32,
    ) -> Result<Vec<KlineData>> {
        let end_time = Utc::now();
        let start_time = end_time - Duration::days(days as i64);

        let mut all_data: Vec<KlineData> = Vec::new();
        let mut current_end = end_time.timestamp_millis();
        let start_ms = start_time.timestamp_millis();

        while current_end > start_ms {
            let klines = self
                .get_klines(symbol, interval, 1000, None, Some(current_end))
                .await?;

            if klines.is_empty() {
                break;
            }

            current_end = klines.first().map(|k| k.timestamp - 1).unwrap_or(0);
            all_data.extend(klines);

            // Rate limiting
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        // Remove duplicates and sort
        all_data.sort_by_key(|k| k.timestamp);
        all_data.dedup_by_key(|k| k.timestamp);

        // Filter to requested date range
        all_data.retain(|k| k.timestamp >= start_ms);

        Ok(all_data)
    }

    /// Fetch klines synchronously (blocking)
    pub fn get_klines_blocking(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<KlineData>> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(self.get_klines(symbol, interval, limit, None, None))
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
    async fn test_client_creation() {
        let client = BybitClient::new();
        assert_eq!(client.base_url, "https://api.bybit.com");
    }
}
