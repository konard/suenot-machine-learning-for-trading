//! Bybit API client for fetching cryptocurrency market data.

use reqwest::Client;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur when interacting with the Bybit API.
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("API returned error: {0}")]
    ApiError(String),

    #[error("Failed to parse response: {0}")]
    ParseError(String),
}

/// A single OHLCV candlestick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Response structure from Bybit kline endpoint.
#[derive(Debug, Deserialize)]
struct KlineResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: KlineResult,
}

#[derive(Debug, Deserialize)]
struct KlineResult {
    list: Vec<Vec<String>>,
}

/// Client for interacting with the Bybit API.
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client.
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch OHLCV kline data from Bybit.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Candle interval ("1", "5", "15", "60", "240", "D")
    /// * `limit` - Number of candles to fetch (max 200)
    ///
    /// # Returns
    ///
    /// Vector of Kline structs ordered from oldest to newest.
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Kline>, BybitError> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url,
            symbol,
            interval,
            limit.min(200)
        );

        let response: KlineResponse = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError(response.ret_msg));
        }

        let mut klines: Vec<Kline> = response
            .result
            .list
            .into_iter()
            .filter_map(|item| {
                if item.len() >= 6 {
                    Some(Kline {
                        timestamp: item[0].parse().ok()?,
                        open: item[1].parse().ok()?,
                        high: item[2].parse().ok()?,
                        low: item[3].parse().ok()?,
                        close: item[4].parse().ok()?,
                        volume: item[5].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by timestamp (oldest first)
        klines.sort_by_key(|k| k.timestamp);

        Ok(klines)
    }

    /// Fetch klines synchronously (blocking).
    pub fn fetch_klines_blocking(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Kline>, BybitError> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url,
            symbol,
            interval,
            limit.min(200)
        );

        let response: KlineResponse = reqwest::blocking::get(&url)?.json()?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError(response.ret_msg));
        }

        let mut klines: Vec<Kline> = response
            .result
            .list
            .into_iter()
            .filter_map(|item| {
                if item.len() >= 6 {
                    Some(Kline {
                        timestamp: item[0].parse().ok()?,
                        open: item[1].parse().ok()?,
                        high: item[2].parse().ok()?,
                        low: item[3].parse().ok()?,
                        close: item[4].parse().ok()?,
                        volume: item[5].parse().ok()?,
                    })
                } else {
                    None
                }
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

/// Generate sample kline data for testing.
pub fn generate_sample_klines(n_samples: usize) -> Vec<Kline> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;

    let mut price = 50000.0;
    let mut klines = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let timestamp = now - ((n_samples - i) as i64 * 3600000); // 1 hour intervals

        // Random walk
        let change = (rand_f64() - 0.5) * 0.04;
        price *= 1.0 + change;

        let high = price * (1.0 + rand_f64().abs() * 0.01);
        let low = price * (1.0 - rand_f64().abs() * 0.01);
        let open = low + (high - low) * rand_f64();
        let close = low + (high - low) * rand_f64();
        let volume = rand_f64() * 1000.0 + 100.0;

        klines.push(Kline {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        });
    }

    klines
}

/// Simple pseudo-random number generator for sample data.
fn rand_f64() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    static mut SEED: u64 = 0;

    unsafe {
        if SEED == 0 {
            SEED = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
        }
        SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1);
        (SEED >> 33) as f64 / (1u64 << 31) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_sample_klines() {
        let klines = generate_sample_klines(100);
        assert_eq!(klines.len(), 100);

        for kline in &klines {
            assert!(kline.high >= kline.low);
            assert!(kline.open >= kline.low && kline.open <= kline.high);
            assert!(kline.close >= kline.low && kline.close <= kline.high);
            assert!(kline.volume > 0.0);
        }
    }
}
