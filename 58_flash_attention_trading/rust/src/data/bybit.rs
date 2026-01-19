//! Bybit API client for cryptocurrency data.
//!
//! Fetches OHLCV data from Bybit's public API.

use super::OhlcvData;
use anyhow::{Context, Result};
use chrono::{TimeZone, Utc};
use serde::{Deserialize, Serialize};

/// Kline/candlestick data from Bybit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub start_time: i64,
    pub open: String,
    pub high: String,
    pub low: String,
    pub close: String,
    pub volume: String,
    pub turnover: String,
}

impl Kline {
    /// Convert to OhlcvData
    pub fn to_ohlcv(&self) -> Result<OhlcvData> {
        Ok(OhlcvData {
            timestamp: Utc.timestamp_millis_opt(self.start_time).unwrap(),
            open: self.open.parse().context("Failed to parse open")?,
            high: self.high.parse().context("Failed to parse high")?,
            low: self.low.parse().context("Failed to parse low")?,
            close: self.close.parse().context("Failed to parse close")?,
            volume: self.volume.parse().context("Failed to parse volume")?,
        })
    }
}

/// Bybit API response wrapper
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
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

/// Bybit API client
pub struct BybitClient {
    base_url: String,
    client: reqwest::blocking::Client,
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
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Create client with custom base URL
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Fetch kline data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Candlestick interval ("1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M")
    /// * `limit` - Number of candles to fetch (max 1000)
    ///
    /// # Example
    /// ```rust,ignore
    /// let client = BybitClient::new();
    /// let klines = client.fetch_klines("BTCUSDT", "60", 100)?;
    /// ```
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<OhlcvData>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url,
            symbol,
            interval,
            limit.min(1000)
        );

        log::info!("Fetching klines from: {}", url);

        let response: BybitResponse = self
            .client
            .get(&url)
            .send()
            .context("Failed to send request")?
            .json()
            .context("Failed to parse response")?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {} - {}", response.ret_code, response.ret_msg);
        }

        let mut ohlcv_data: Vec<OhlcvData> = response
            .result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() >= 6 {
                    Some(OhlcvData {
                        timestamp: Utc
                            .timestamp_millis_opt(item[0].parse().unwrap_or(0))
                            .unwrap(),
                        open: item[1].parse().unwrap_or(0.0),
                        high: item[2].parse().unwrap_or(0.0),
                        low: item[3].parse().unwrap_or(0.0),
                        close: item[4].parse().unwrap_or(0.0),
                        volume: item[5].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first, we want oldest first
        ohlcv_data.reverse();

        log::info!("Fetched {} klines for {}", ohlcv_data.len(), symbol);

        Ok(ohlcv_data)
    }

    /// Fetch klines with pagination for longer history
    pub fn fetch_klines_extended(
        &self,
        symbol: &str,
        interval: &str,
        total: usize,
    ) -> Result<Vec<OhlcvData>> {
        let mut all_data = Vec::new();
        let mut end_time: Option<i64> = None;

        while all_data.len() < total {
            let limit = (total - all_data.len()).min(1000);

            let url = match end_time {
                Some(et) => format!(
                    "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}&end={}",
                    self.base_url, symbol, interval, limit, et
                ),
                None => format!(
                    "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
                    self.base_url, symbol, interval, limit
                ),
            };

            let response: BybitResponse = self
                .client
                .get(&url)
                .send()
                .context("Failed to send request")?
                .json()
                .context("Failed to parse response")?;

            if response.ret_code != 0 {
                anyhow::bail!("Bybit API error: {}", response.ret_msg);
            }

            if response.result.list.is_empty() {
                break;
            }

            let batch: Vec<OhlcvData> = response
                .result
                .list
                .iter()
                .filter_map(|item| {
                    if item.len() >= 6 {
                        Some(OhlcvData {
                            timestamp: Utc
                                .timestamp_millis_opt(item[0].parse().unwrap_or(0))
                                .unwrap(),
                            open: item[1].parse().unwrap_or(0.0),
                            high: item[2].parse().unwrap_or(0.0),
                            low: item[3].parse().unwrap_or(0.0),
                            close: item[4].parse().unwrap_or(0.0),
                            volume: item[5].parse().unwrap_or(0.0),
                        })
                    } else {
                        None
                    }
                })
                .collect();

            // Get the earliest timestamp for next request
            if let Some(last) = batch.last() {
                end_time = Some(last.timestamp.timestamp_millis() - 1);
            } else {
                break;
            }

            all_data.extend(batch);

            // Rate limiting
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        // Reverse to get oldest first
        all_data.reverse();

        Ok(all_data)
    }
}

/// Convenience function to fetch Bybit klines
pub fn fetch_bybit_klines(symbol: &str, interval: &str, limit: usize) -> Result<Vec<OhlcvData>> {
    let client = BybitClient::new();
    client.fetch_klines(symbol, interval, limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_to_ohlcv() {
        let kline = Kline {
            start_time: 1704067200000,
            open: "42000.5".to_string(),
            high: "42500.0".to_string(),
            low: "41500.0".to_string(),
            close: "42100.0".to_string(),
            volume: "100.5".to_string(),
            turnover: "4215052.5".to_string(),
        };

        let ohlcv = kline.to_ohlcv().unwrap();
        assert!((ohlcv.open - 42000.5).abs() < 0.01);
        assert!((ohlcv.close - 42100.0).abs() < 0.01);
    }
}
