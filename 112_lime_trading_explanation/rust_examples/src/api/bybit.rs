//! Bybit API client for fetching cryptocurrency market data.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// OHLCV candle data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Bybit API response structure
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

/// Bybit API client
pub struct BybitClient {
    base_url: String,
    client: reqwest::blocking::Client,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Fetch kline (candlestick) data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Candle interval ("1", "5", "15", "60", "240", "D")
    /// * `limit` - Number of candles to fetch (max 1000)
    ///
    /// # Returns
    /// Vector of Candle structs sorted by timestamp ascending
    pub fn fetch_klines(&self, symbol: &str, interval: &str, limit: usize) -> Result<Vec<Candle>> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", "spot"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.to_string()),
            ])
            .send()?;

        let data: BybitResponse = response.json()?;

        if data.ret_code != 0 {
            return Err(anyhow!("Bybit API error: {}", data.ret_msg));
        }

        let mut candles: Vec<Candle> = data
            .result
            .list
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 6 {
                    let timestamp_ms: i64 = row[0].parse().ok()?;
                    let timestamp = DateTime::from_timestamp_millis(timestamp_ms)?;

                    Some(Candle {
                        timestamp,
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

        // Sort by timestamp ascending
        candles.sort_by_key(|c| c.timestamp);

        Ok(candles)
    }

    /// Fetch multiple symbols
    pub fn fetch_multiple_symbols(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: usize,
    ) -> Result<std::collections::HashMap<String, Vec<Candle>>> {
        let mut result = std::collections::HashMap::new();

        for symbol in symbols {
            match self.fetch_klines(symbol, interval, limit) {
                Ok(candles) => {
                    result.insert(symbol.to_string(), candles);
                }
                Err(e) => {
                    log::warn!("Failed to fetch {}: {}", symbol, e);
                }
            }
        }

        Ok(result)
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
        assert!(!client.base_url.is_empty());
    }
}
