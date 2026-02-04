//! Bybit API client implementation

use anyhow::{Context, Result};
use chrono::{TimeZone, Utc};
use reqwest::Client;

use super::types::{ApiResponse, Kline, KlineData, KlinesResult, Ticker, TickersResult};

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

    /// Create a new client with custom base URL
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Fetch klines/candlestick data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Kline interval ("1", "5", "15", "60", "240", "D", "W")
    /// * `limit` - Number of klines to fetch (max 1000)
    ///
    /// # Returns
    /// KlineData containing the fetched klines
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<KlineData> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.min(1000).to_string()),
            ])
            .send()
            .await
            .context("Failed to send request")?;

        let api_response: ApiResponse<KlinesResult> = response
            .json()
            .await
            .context("Failed to parse response")?;

        if api_response.ret_code != 0 {
            anyhow::bail!("API error: {}", api_response.ret_msg);
        }

        let klines = self.parse_klines(&api_response.result.list)?;

        Ok(KlineData::new(
            symbol.to_string(),
            interval.to_string(),
            klines,
        ))
    }

    /// Fetch klines synchronously (blocking)
    pub fn get_klines_blocking(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<KlineData> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let client = reqwest::blocking::Client::new();
        let response = client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.min(1000).to_string()),
            ])
            .send()
            .context("Failed to send request")?;

        let api_response: ApiResponse<KlinesResult> = response
            .json()
            .context("Failed to parse response")?;

        if api_response.ret_code != 0 {
            anyhow::bail!("API error: {}", api_response.ret_msg);
        }

        let klines = self.parse_klines(&api_response.result.list)?;

        Ok(KlineData::new(
            symbol.to_string(),
            interval.to_string(),
            klines,
        ))
    }

    /// Parse raw kline data from API
    fn parse_klines(&self, raw: &[Vec<String>]) -> Result<Vec<Kline>> {
        let mut klines: Vec<Kline> = raw
            .iter()
            .filter_map(|row| {
                if row.len() < 7 {
                    return None;
                }

                let timestamp_ms: i64 = row[0].parse().ok()?;
                let timestamp = Utc.timestamp_millis_opt(timestamp_ms).single()?;

                Some(Kline {
                    timestamp,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
                    turnover: row[6].parse().ok()?,
                })
            })
            .collect();

        // Bybit returns descending order, we want ascending
        klines.sort_by_key(|k| k.timestamp);

        Ok(klines)
    }

    /// Fetch all tickers
    pub async fn get_tickers(&self, category: &str) -> Result<Vec<Ticker>> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[("category", category)])
            .send()
            .await
            .context("Failed to send request")?;

        let api_response: ApiResponse<TickersResult> = response
            .json()
            .await
            .context("Failed to parse response")?;

        if api_response.ret_code != 0 {
            anyhow::bail!("API error: {}", api_response.ret_msg);
        }

        Ok(api_response.result.list)
    }

    /// Fetch data for multiple timeframes
    pub async fn get_multi_timeframe(
        &self,
        symbol: &str,
        intervals: &[&str],
        limit: usize,
    ) -> Result<Vec<KlineData>> {
        let mut results = Vec::new();

        for interval in intervals {
            let data = self.get_klines(symbol, interval, limit).await?;
            results.push(data);
            // Small delay to avoid rate limiting
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        Ok(results)
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
    fn test_client_creation() {
        let client = BybitClient::new();
        assert_eq!(client.base_url, "https://api.bybit.com");
    }

    #[test]
    fn test_custom_base_url() {
        let client = BybitClient::with_base_url("https://test.bybit.com");
        assert_eq!(client.base_url, "https://test.bybit.com");
    }
}
