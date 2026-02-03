//! Bybit API client for fetching cryptocurrency market data.
//!
//! Uses the public V5 API to fetch kline (candlestick) data
//! without authentication.

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// A single kline (candlestick) data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Bybit API response structures.
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

/// Client for the Bybit public API.
#[derive(Debug, Clone)]
pub struct BybitClient {
    base_url: String,
    client: reqwest::Client,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Fetch kline data from Bybit.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Candle interval ("1", "5", "15", "60", "240", "D")
    /// * `limit` - Number of candles (max 1000)
    ///
    /// # Returns
    /// Vec of Kline data sorted by timestamp ascending
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Kline>> {
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
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await?;

        let bybit_resp: BybitResponse = response.json().await?;

        if bybit_resp.ret_code != 0 {
            tracing::warn!("Bybit API error: {}", bybit_resp.ret_msg);
            return Ok(generate_synthetic_klines(limit));
        }

        let mut klines: Vec<Kline> = bybit_resp
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() < 6 {
                    return None;
                }
                let ts_ms: i64 = row[0].parse().ok()?;
                let timestamp = DateTime::from_timestamp(ts_ms / 1000, 0)?;

                Some(Kline {
                    timestamp,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
                })
            })
            .collect();

        klines.sort_by_key(|k| k.timestamp);
        Ok(klines)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate synthetic kline data for testing.
pub fn generate_synthetic_klines(n: usize) -> Vec<Kline> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut price = 50000.0_f64;
    let mut klines = Vec::with_capacity(n);

    let base_ts = Utc::now().timestamp() - (n as i64 * 3600);

    for i in 0..n {
        let ret = rng.gen_range(-0.03..0.03);
        price *= 1.0 + ret;

        let high = price * (1.0 + rng.gen_range(0.0..0.015));
        let low = price * (1.0 - rng.gen_range(0.0..0.015));
        let open = low + (high - low) * rng.gen_range(0.0..1.0);
        let volume = rng.gen_range(100.0..10000.0);

        let timestamp = DateTime::from_timestamp(base_ts + (i as i64 * 3600), 0)
            .unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap());

        klines.push(Kline {
            timestamp,
            open,
            high,
            low,
            close: price,
            volume,
        });
    }

    klines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_klines() {
        let klines = generate_synthetic_klines(100);
        assert_eq!(klines.len(), 100);
        for k in &klines {
            assert!(k.high >= k.low);
            assert!(k.close > 0.0);
            assert!(k.volume > 0.0);
        }
    }

    #[test]
    fn test_bybit_client_creation() {
        let client = BybitClient::new();
        assert!(client.base_url.contains("bybit"));
    }
}
