//! Bybit API client for fetching cryptocurrency data.

use anyhow::{Context, Result};
use chrono::Utc;
use reqwest::Client;
use serde::Deserialize;
use tracing::{debug, info};

use super::ohlcv::{MultiSymbolData, OHLCV};

const BASE_URL: &str = "https://api.bybit.com/v5/market/kline";

/// Response from Bybit API.
#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Client for fetching data from Bybit exchange.
pub struct BybitClient {
    client: Client,
}

impl BybitClient {
    /// Create a new Bybit client.
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }

    /// Fetch kline/candlestick data.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Kline interval (1, 5, 15, 30, 60, 120, 240, D, W)
    /// * `limit` - Number of records (max 1000)
    /// * `end_time` - End timestamp in milliseconds
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
        end_time: Option<i64>,
    ) -> Result<Vec<OHLCV>> {
        let mut params = vec![
            ("category", "linear".to_string()),
            ("symbol", symbol.to_string()),
            ("interval", interval.to_string()),
            ("limit", limit.to_string()),
        ];

        if let Some(end) = end_time {
            params.push(("end", end.to_string()));
        }

        let response = self
            .client
            .get(BASE_URL)
            .query(&params)
            .send()
            .await
            .context("Failed to send request to Bybit")?;

        let data: BybitResponse = response
            .json()
            .await
            .context("Failed to parse Bybit response")?;

        if data.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", data.ret_msg);
        }

        let ohlcv_data: Vec<OHLCV> = data
            .result
            .list
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 6 {
                    Some(OHLCV {
                        timestamp: row[0].parse().ok()?,
                        open: row[1].parse().ok()?,
                        high: row[2].parse().ok()?,
                        low: row[3].parse().ok()?,
                        close: row[4].parse().ok()?,
                        volume: row[5].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(ohlcv_data)
    }

    /// Fetch historical data for specified number of days.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Kline interval
    /// * `days` - Number of days to fetch
    pub async fn fetch_historical_klines(
        &self,
        symbol: &str,
        interval: &str,
        days: u32,
    ) -> Result<Vec<OHLCV>> {
        let mut all_data = Vec::new();
        let mut end_time = Utc::now().timestamp_millis();

        // Calculate how many candles we need
        let interval_minutes: u32 = match interval {
            "1" => 1,
            "5" => 5,
            "15" => 15,
            "30" => 30,
            "60" => 60,
            "120" => 120,
            "240" => 240,
            "D" => 1440,
            "W" => 10080,
            _ => 60,
        };

        let total_candles = (days * 24 * 60) / interval_minutes;
        info!(
            "Fetching {} candles for {} days of {}",
            total_candles, days, symbol
        );

        while all_data.len() < total_candles as usize {
            let batch = self
                .fetch_klines(symbol, interval, 1000, Some(end_time))
                .await?;

            if batch.is_empty() {
                break;
            }

            debug!("Fetched {} candles for {}", batch.len(), symbol);

            // Update end_time for next batch
            if let Some(last) = batch.last() {
                end_time = last.timestamp - 1;
            }

            all_data.extend(batch);

            // Small delay to avoid rate limiting
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        // Sort by timestamp ascending
        all_data.sort_by_key(|k| k.timestamp);

        // Remove duplicates
        all_data.dedup_by_key(|k| k.timestamp);

        info!("Total candles fetched for {}: {}", symbol, all_data.len());

        Ok(all_data)
    }

    /// Fetch data for multiple symbols.
    ///
    /// # Arguments
    /// * `symbols` - List of trading pairs
    /// * `interval` - Kline interval
    /// * `days` - Number of days to fetch
    pub async fn fetch_multi_symbol(
        &self,
        symbols: &[&str],
        interval: &str,
        days: u32,
    ) -> Result<MultiSymbolData> {
        let mut multi_data = MultiSymbolData::new();

        for symbol in symbols {
            info!("Fetching data for {}", symbol);
            let data = self.fetch_historical_klines(symbol, interval, days).await?;
            multi_data.add_symbol(symbol.to_string(), data);
        }

        Ok(multi_data)
    }

    /// Get available symbols from Bybit.
    pub async fn get_available_symbols(&self) -> Result<Vec<String>> {
        let response = self
            .client
            .get("https://api.bybit.com/v5/market/instruments-info")
            .query(&[("category", "linear")])
            .send()
            .await
            .context("Failed to fetch instruments")?;

        #[derive(Deserialize)]
        struct InstrumentResponse {
            result: InstrumentResult,
        }

        #[derive(Deserialize)]
        struct InstrumentResult {
            list: Vec<Instrument>,
        }

        #[derive(Deserialize)]
        struct Instrument {
            symbol: String,
        }

        let data: InstrumentResponse = response.json().await?;
        let symbols: Vec<String> = data.result.list.into_iter().map(|i| i.symbol).collect();

        Ok(symbols)
    }

    /// Get top symbols by volume.
    pub async fn get_top_symbols_by_volume(&self, limit: usize) -> Result<Vec<String>> {
        let response = self
            .client
            .get("https://api.bybit.com/v5/market/tickers")
            .query(&[("category", "linear")])
            .send()
            .await
            .context("Failed to fetch tickers")?;

        #[derive(Deserialize)]
        struct TickerResponse {
            result: TickerResult,
        }

        #[derive(Deserialize)]
        struct TickerResult {
            list: Vec<Ticker>,
        }

        #[derive(Deserialize)]
        struct Ticker {
            symbol: String,
            #[serde(rename = "volume24h")]
            volume_24h: String,
        }

        let data: TickerResponse = response.json().await?;

        let mut tickers: Vec<(String, f64)> = data
            .result
            .list
            .into_iter()
            .filter_map(|t| {
                let volume: f64 = t.volume_24h.parse().ok()?;
                // Filter for USDT pairs only
                if t.symbol.ends_with("USDT") {
                    Some((t.symbol, volume))
                } else {
                    None
                }
            })
            .collect();

        // Sort by volume descending
        tickers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top N
        let top_symbols: Vec<String> = tickers.into_iter().take(limit).map(|(s, _)| s).collect();

        Ok(top_symbols)
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
    async fn test_fetch_klines() {
        let client = BybitClient::new();
        let result = client.fetch_klines("BTCUSDT", "60", 10, None).await;

        // This test may fail without network access
        if let Ok(data) = result {
            assert!(!data.is_empty());
            assert!(data[0].close > 0.0);
        }
    }
}
