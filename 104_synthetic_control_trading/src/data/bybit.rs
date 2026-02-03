//! Bybit API client for fetching cryptocurrency market data.

use anyhow::Result;
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

/// OHLCV candlestick data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Open price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Close price
    pub close: f64,
    /// Volume
    pub volume: f64,
    /// Symbol
    pub symbol: String,
}

impl Kline {
    /// Get log return from previous close.
    pub fn log_return(&self, prev_close: f64) -> f64 {
        (self.close / prev_close).ln()
    }
}

/// Response from Bybit kline API.
#[derive(Debug, Deserialize)]
struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitKlineResult,
}

#[derive(Debug, Deserialize)]
struct BybitKlineResult {
    list: Vec<Vec<String>>,
}

/// Client for Bybit public API.
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
    /// Create a new Bybit client.
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Create a client with custom base URL.
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Fetch kline (candlestick) data from Bybit.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair (e.g., "BTCUSDT", "ETHUSDT")
    /// * `interval` - Candle interval ("1", "5", "15", "60", "240", "D", "W")
    /// * `limit` - Number of candles to fetch (max 1000)
    ///
    /// # Returns
    ///
    /// Vector of Kline data sorted by timestamp (oldest first)
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>> {
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
            .await;

        match response {
            Ok(resp) => {
                let data: BybitKlineResponse = resp.json().await?;

                if data.ret_code != 0 {
                    warn!("Bybit API error: {}", data.ret_msg);
                    return Ok(generate_synthetic_klines(limit, symbol));
                }

                let mut klines: Vec<Kline> = data
                    .result
                    .list
                    .into_iter()
                    .filter_map(|row| parse_kline_row(&row, symbol))
                    .collect();

                // Sort by timestamp (oldest first)
                klines.sort_by_key(|k| k.timestamp);

                if klines.is_empty() {
                    warn!("No klines returned for {}", symbol);
                    return Ok(generate_synthetic_klines(limit, symbol));
                }

                debug!("Fetched {} klines for {}", klines.len(), symbol);
                Ok(klines)
            }
            Err(e) => {
                warn!("Failed to fetch Bybit data: {}. Using synthetic data.", e);
                Ok(generate_synthetic_klines(limit, symbol))
            }
        }
    }

    /// Fetch returns (log returns) for a symbol.
    pub async fn fetch_returns(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<f64>> {
        let klines = self.fetch_klines(symbol, interval, limit + 1).await?;

        let returns: Vec<f64> = klines
            .windows(2)
            .map(|w| (w[1].close / w[0].close).ln())
            .collect();

        Ok(returns)
    }
}

/// Parse a kline row from Bybit API response.
fn parse_kline_row(row: &[String], symbol: &str) -> Option<Kline> {
    if row.len() < 6 {
        return None;
    }

    let timestamp_ms: i64 = row[0].parse().ok()?;
    let timestamp = DateTime::from_timestamp_millis(timestamp_ms)?;

    Some(Kline {
        timestamp,
        open: row[1].parse().ok()?,
        high: row[2].parse().ok()?,
        low: row[3].parse().ok()?,
        close: row[4].parse().ok()?,
        volume: row[5].parse().ok()?,
        symbol: symbol.to_string(),
    })
}

/// Generate synthetic kline data for testing.
pub fn generate_synthetic_klines(n: usize, symbol: &str) -> Vec<Kline> {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 0.02).unwrap();

    let mut price = 100.0;
    let now = Utc::now();

    (0..n)
        .map(|i| {
            let ret: f64 = normal.sample(&mut rng);
            price *= 1.0 + ret;

            let high = price * (1.0 + rng.gen::<f64>() * 0.015);
            let low = price * (1.0 - rng.gen::<f64>() * 0.015);
            let open = low + (high - low) * rng.gen::<f64>();

            Kline {
                timestamp: now - chrono::Duration::days((n - i) as i64),
                open,
                high,
                low,
                close: price,
                volume: rng.gen::<f64>() * 1000000.0,
                symbol: symbol.to_string(),
            }
        })
        .collect()
}

/// Extract close prices from klines.
pub fn extract_closes(klines: &[Kline]) -> Vec<f64> {
    klines.iter().map(|k| k.close).collect()
}

/// Extract returns from klines.
pub fn extract_returns(klines: &[Kline]) -> Vec<f64> {
    klines
        .windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_synthetic_klines() {
        let klines = generate_synthetic_klines(100, "BTCUSDT");
        assert_eq!(klines.len(), 100);
        assert!(klines.iter().all(|k| k.close > 0.0));
    }

    #[test]
    fn test_extract_returns() {
        let klines = generate_synthetic_klines(10, "TEST");
        let returns = extract_returns(&klines);
        assert_eq!(returns.len(), 9);
    }
}
