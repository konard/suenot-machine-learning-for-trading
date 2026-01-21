//! Bybit API client for market data.

use crate::{Result, ZeroShotError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// OHLCV candle data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub turnover: f64,
}

/// Bybit API response wrapper.
#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: T,
}

/// Kline result from Bybit API.
#[derive(Debug, Deserialize)]
struct KlineResult {
    list: Vec<Vec<String>>,
}

/// Ticker result from Bybit API.
#[derive(Debug, Deserialize)]
struct TickerResult {
    list: Vec<TickerInfo>,
}

/// Ticker information.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TickerInfo {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    #[serde(rename = "turnover24h")]
    pub turnover_24h: String,
    #[serde(rename = "highPrice24h")]
    pub high_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_24h: String,
}

/// Bybit API client.
#[derive(Debug, Clone)]
pub struct BybitClient {
    base_url: String,
    client: reqwest::Client,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Create a new Bybit client.
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Create a client with custom base URL.
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Fetch kline (candlestick) data.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Interval (e.g., "1h", "15m", "1d")
    /// * `limit` - Number of candles to fetch (max 200)
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Kline>> {
        let endpoint = format!("{}/v5/market/kline", self.base_url);

        // Convert interval to Bybit format
        let interval_minutes = match interval {
            "1m" => "1",
            "3m" => "3",
            "5m" => "5",
            "15m" => "15",
            "30m" => "30",
            "1h" | "60" => "60",
            "2h" => "120",
            "4h" => "240",
            "6h" => "360",
            "12h" => "720",
            "1d" | "D" => "D",
            "1w" | "W" => "W",
            other => other,
        };

        let response = self
            .client
            .get(&endpoint)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("interval", interval_minutes),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await
            .map_err(|e| ZeroShotError::ApiError(e.to_string()))?;

        let data: BybitResponse<KlineResult> = response
            .json()
            .await
            .map_err(|e| ZeroShotError::ApiError(e.to_string()))?;

        if data.ret_code != 0 {
            return Err(ZeroShotError::ApiError(data.ret_msg));
        }

        // Parse klines
        let mut klines: Vec<Kline> = data
            .result
            .list
            .into_iter()
            .filter_map(|row| {
                if row.len() < 7 {
                    return None;
                }
                let timestamp_ms: i64 = row[0].parse().ok()?;
                let timestamp =
                    DateTime::from_timestamp_millis(timestamp_ms)?;
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

        // Sort by timestamp ascending
        klines.sort_by_key(|k| k.timestamp);

        Ok(klines)
    }

    /// Fetch ticker information.
    pub async fn fetch_ticker(&self, symbol: &str) -> Result<TickerInfo> {
        let endpoint = format!("{}/v5/market/tickers", self.base_url);

        let response = self
            .client
            .get(&endpoint)
            .query(&[("category", "linear"), ("symbol", symbol)])
            .send()
            .await
            .map_err(|e| ZeroShotError::ApiError(e.to_string()))?;

        let data: BybitResponse<TickerResult> = response
            .json()
            .await
            .map_err(|e| ZeroShotError::ApiError(e.to_string()))?;

        if data.ret_code != 0 {
            return Err(ZeroShotError::ApiError(data.ret_msg));
        }

        data.result
            .list
            .into_iter()
            .next()
            .ok_or_else(|| ZeroShotError::ApiError("Ticker not found".into()))
    }

    /// Fetch multiple symbols' klines concurrently.
    pub async fn fetch_multiple_klines(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: u32,
    ) -> Result<Vec<(String, Vec<Kline>)>> {
        let mut results = Vec::new();

        for symbol in symbols {
            match self.fetch_klines(symbol, interval, limit).await {
                Ok(klines) => results.push((symbol.to_string(), klines)),
                Err(e) => {
                    tracing::warn!("Failed to fetch {}: {}", symbol, e);
                }
            }
        }

        Ok(results)
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
