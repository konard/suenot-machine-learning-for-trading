//! Bybit API client

use anyhow::{anyhow, Result};
use reqwest::Client;
use std::time::Duration;

use super::types::{ApiResponse, Kline, KlinesResult, Ticker, TickersResult};

/// Bybit REST API client
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        Self::with_base_url("https://api.bybit.com")
    }

    /// Create client with custom base URL
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

    /// Create testnet client
    pub fn testnet() -> Self {
        Self::with_base_url("https://api-testnet.bybit.com")
    }

    /// Fetch kline (candlestick) data
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "API request failed with status: {}",
                response.status()
            ));
        }

        let api_response: ApiResponse<KlinesResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(anyhow!(
                "API error: {} - {}",
                api_response.ret_code,
                api_response.ret_msg
            ));
        }

        // Parse klines from string arrays
        let klines: Vec<Kline> = api_response
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() >= 7 {
                    Some(Kline::new(
                        row[0].parse().ok()?,
                        row[1].parse().ok()?,
                        row[2].parse().ok()?,
                        row[3].parse().ok()?,
                        row[4].parse().ok()?,
                        row[5].parse().ok()?,
                        row[6].parse().ok()?,
                    ))
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first, reverse to get chronological order
        let mut klines = klines;
        klines.reverse();

        Ok(klines)
    }

    /// Fetch klines with start time
    pub async fn get_klines_from(
        &self,
        symbol: &str,
        interval: &str,
        start_time: i64,
        limit: u32,
    ) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&start={}&limit={}",
            self.base_url, symbol, interval, start_time, limit
        );

        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "API request failed with status: {}",
                response.status()
            ));
        }

        let api_response: ApiResponse<KlinesResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(anyhow!(
                "API error: {} - {}",
                api_response.ret_code,
                api_response.ret_msg
            ));
        }

        let klines: Vec<Kline> = api_response
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() >= 7 {
                    Some(Kline::new(
                        row[0].parse().ok()?,
                        row[1].parse().ok()?,
                        row[2].parse().ok()?,
                        row[3].parse().ok()?,
                        row[4].parse().ok()?,
                        row[5].parse().ok()?,
                        row[6].parse().ok()?,
                    ))
                } else {
                    None
                }
            })
            .collect();

        let mut klines = klines;
        klines.reverse();

        Ok(klines)
    }

    /// Fetch ticker for a symbol
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        let url = format!(
            "{}/v5/market/tickers?category=linear&symbol={}",
            self.base_url, symbol
        );

        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "API request failed with status: {}",
                response.status()
            ));
        }

        let api_response: ApiResponse<TickersResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(anyhow!(
                "API error: {} - {}",
                api_response.ret_code,
                api_response.ret_msg
            ));
        }

        api_response
            .result
            .list
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No ticker found for symbol: {}", symbol))
    }

    /// Fetch tickers for all perpetual contracts
    pub async fn get_all_tickers(&self) -> Result<Vec<Ticker>> {
        let url = format!(
            "{}/v5/market/tickers?category=linear",
            self.base_url
        );

        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "API request failed with status: {}",
                response.status()
            ));
        }

        let api_response: ApiResponse<TickersResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(anyhow!(
                "API error: {} - {}",
                api_response.ret_code,
                api_response.ret_msg
            ));
        }

        Ok(api_response.result.list)
    }

    /// Fetch historical klines (multiple API calls for large date ranges)
    pub async fn get_historical_klines(
        &self,
        symbol: &str,
        interval: &str,
        days: u32,
    ) -> Result<Vec<Kline>> {
        let interval_ms = Self::interval_to_ms(interval);
        let candles_per_day = 86_400_000 / interval_ms;
        let total_candles = (candles_per_day * days as i64) as usize;

        let mut all_klines = Vec::with_capacity(total_candles);
        let now = chrono::Utc::now().timestamp_millis();
        let start_time = now - (days as i64 * 86_400_000);

        let mut current_start = start_time;
        let limit = 1000u32;

        while current_start < now {
            let klines = self
                .get_klines_from(symbol, interval, current_start, limit)
                .await?;

            if klines.is_empty() {
                break;
            }

            let last_timestamp = klines.last().map(|k| k.timestamp).unwrap_or(now);
            all_klines.extend(klines);

            current_start = last_timestamp + interval_ms;

            // Rate limiting
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Remove duplicates and sort
        all_klines.sort_by_key(|k| k.timestamp);
        all_klines.dedup_by_key(|k| k.timestamp);

        Ok(all_klines)
    }

    /// Convert interval string to milliseconds
    fn interval_to_ms(interval: &str) -> i64 {
        match interval {
            "1" => 60_000,
            "3" => 180_000,
            "5" => 300_000,
            "15" => 900_000,
            "30" => 1_800_000,
            "60" => 3_600_000,
            "120" => 7_200_000,
            "240" => 14_400_000,
            "360" => 21_600_000,
            "720" => 43_200_000,
            "D" => 86_400_000,
            "W" => 604_800_000,
            _ => 3_600_000, // Default to 1 hour
        }
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
        assert!(result.is_ok());

        let ticker = result.unwrap();
        assert_eq!(ticker.symbol, "BTCUSDT");
    }
}
