//! Bybit API client for fetching cryptocurrency kline data.
//!
//! Uses the Bybit V5 public REST API. Falls back to synthetic data
//! generation when the API is unavailable.

use chrono::{DateTime, Utc};
use rand::Rng;
use rand_distr::Normal;
use reqwest::Client;
use serde::Deserialize;
use tracing::{info, warn};

/// A single kline (candlestick) data point.
#[derive(Debug, Clone)]
pub struct Kline {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

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

/// Bybit API client for fetching kline data.
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client.
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api.bybit.com/v5/market/kline".to_string(),
        }
    }

    /// Fetch kline data for a symbol.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Candle interval ("1", "5", "15", "60", "240", "D", "W")
    /// * `limit` - Number of candles (max 1000)
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Kline>> {
        let limit = limit.min(1000);

        let response = self
            .client
            .get(&self.base_url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await;

        match response {
            Ok(resp) => {
                let body: BybitResponse = resp.json().await?;
                if body.ret_code != 0 {
                    warn!("Bybit API error: {}. Using synthetic data.", body.ret_msg);
                    return Ok(generate_synthetic_klines(limit, symbol));
                }

                let mut klines = Vec::with_capacity(body.result.list.len());
                for row in &body.result.list {
                    if row.len() < 6 {
                        continue;
                    }
                    let ts_ms: i64 = row[0].parse()?;
                    let timestamp =
                        DateTime::from_timestamp_millis(ts_ms).unwrap_or_else(Utc::now);
                    klines.push(Kline {
                        timestamp,
                        open: row[1].parse()?,
                        high: row[2].parse()?,
                        low: row[3].parse()?,
                        close: row[4].parse()?,
                        volume: row[5].parse()?,
                    });
                }
                klines.reverse(); // Bybit returns newest first
                info!("Fetched {} klines for {} from Bybit", klines.len(), symbol);
                Ok(klines)
            }
            Err(e) => {
                warn!("Bybit API request failed: {}. Using synthetic data.", e);
                Ok(generate_synthetic_klines(limit, symbol))
            }
        }
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate synthetic kline data for testing.
pub fn generate_synthetic_klines(n: usize, symbol: &str) -> Vec<Kline> {
    info!("Generating {} synthetic klines for {}", n, symbol);
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0002, 0.03).unwrap();

    let base_price = match symbol {
        "BTCUSDT" => 50000.0,
        "ETHUSDT" => 3000.0,
        "SOLUSDT" => 100.0,
        _ => 1000.0,
    };

    let mut klines = Vec::with_capacity(n);
    let mut price = base_price;

    for i in 0..n {
        let ret: f64 = rng.sample(normal);
        price *= 1.0 + ret;

        let spread_normal = Normal::new(0.0, 0.015).unwrap();
        let high = price * (1.0 + rng.sample::<f64, _>(spread_normal).abs());
        let low = price * (1.0 - rng.sample::<f64, _>(spread_normal).abs());
        let open = low + rng.gen::<f64>() * (high - low);
        let volume = rng.gen_range(1_000_000.0..100_000_000.0);

        let timestamp = Utc::now() - chrono::Duration::days((n - i) as i64);

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

/// Extract close-to-close returns from klines.
pub fn extract_returns(klines: &[Kline]) -> Vec<f64> {
    if klines.len() < 2 {
        return Vec::new();
    }
    klines
        .windows(2)
        .map(|w| (w[1].close - w[0].close) / w[0].close)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_klines() {
        let klines = generate_synthetic_klines(100, "BTCUSDT");
        assert_eq!(klines.len(), 100);
        for k in &klines {
            assert!(k.high >= k.low);
            assert!(k.close > 0.0);
            assert!(k.volume > 0.0);
        }
    }

    #[test]
    fn test_extract_returns() {
        let klines = generate_synthetic_klines(50, "ETHUSDT");
        let returns = extract_returns(&klines);
        assert_eq!(returns.len(), 49);
    }
}
