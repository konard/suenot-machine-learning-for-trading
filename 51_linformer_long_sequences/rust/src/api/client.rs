//! HTTP client for Bybit API.

use reqwest::Client;
use crate::api::types::{ApiError, Kline, KlineResponse};

/// Bybit API client for fetching market data.
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
    /// Create a new Bybit client with default settings.
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Create a new Bybit client with custom base URL.
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
        }
    }

    /// Fetch kline (candlestick) data from Bybit.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Kline interval (e.g., "1", "5", "15", "60", "D")
    /// * `limit` - Number of klines to fetch (max 1000)
    /// * `start` - Optional start timestamp in milliseconds
    /// * `end` - Optional end timestamp in milliseconds
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
        start: Option<u64>,
        end: Option<u64>,
    ) -> Result<Vec<Kline>, ApiError> {
        let mut url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit.min(1000)
        );

        if let Some(start_time) = start {
            url.push_str(&format!("&start={}", start_time));
        }
        if let Some(end_time) = end {
            url.push_str(&format!("&end={}", end_time));
        }

        let response: KlineResponse = self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(ApiError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let klines: Result<Vec<Kline>, ApiError> = response
            .result
            .list
            .iter()
            .map(|item| Kline::from_bybit_list(item))
            .collect();

        // Bybit returns data in descending order, reverse to ascending
        let mut klines = klines?;
        klines.reverse();

        Ok(klines)
    }

    /// Fetch multiple batches of kline data for long sequences.
    /// This handles pagination to get more than 1000 klines.
    pub async fn get_klines_long(
        &self,
        symbol: &str,
        interval: &str,
        total_count: usize,
    ) -> Result<Vec<Kline>, ApiError> {
        let mut all_klines = Vec::with_capacity(total_count);
        let mut end_time: Option<u64> = None;

        while all_klines.len() < total_count {
            let remaining = total_count - all_klines.len();
            let batch_size = remaining.min(1000) as u32;

            let batch = self
                .get_klines(symbol, interval, batch_size, None, end_time)
                .await?;

            if batch.is_empty() {
                break;
            }

            // Get the earliest timestamp for next batch
            if let Some(first) = batch.first() {
                end_time = Some(first.start_time - 1);
            }

            // Prepend batch to maintain chronological order
            let mut new_klines = batch;
            new_klines.append(&mut all_klines);
            all_klines = new_klines;
        }

        // Trim to exact requested count
        if all_klines.len() > total_count {
            all_klines = all_klines[all_klines.len() - total_count..].to_vec();
        }

        Ok(all_klines)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = BybitClient::new();
        assert_eq!(client.base_url, "https://api.bybit.com");
    }

    #[test]
    fn test_client_with_custom_url() {
        let client = BybitClient::with_base_url("https://api-testnet.bybit.com");
        assert_eq!(client.base_url, "https://api-testnet.bybit.com");
    }
}
