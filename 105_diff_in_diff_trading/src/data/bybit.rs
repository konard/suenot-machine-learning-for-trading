//! Bybit API client for fetching cryptocurrency market data.

use chrono::{DateTime, Utc};
use serde::Deserialize;

/// A single candlestick (OHLCV) data point.
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitResult,
}

#[derive(Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Client for the Bybit public market data API.
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

    /// Fetch kline (candlestick) data for a symbol.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair, e.g., "BTCUSDT"
    /// * `interval` - Kline interval: "1", "5", "15", "60", "D", "W"
    /// * `limit` - Number of candles (max 1000)
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>, Box<dyn std::error::Error>> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", "spot"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.min(1000).to_string()),
            ])
            .send()
            .await?;

        let data: BybitResponse = response.json().await?;

        if data.ret_code != 0 {
            return Err(format!("Bybit API error: {}", data.ret_msg).into());
        }

        let mut candles: Vec<Candle> = data
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() < 6 {
                    return None;
                }
                let ts_ms: i64 = row[0].parse().ok()?;
                let timestamp =
                    DateTime::from_timestamp(ts_ms / 1000, ((ts_ms % 1000) * 1_000_000) as u32)?;
                Some(Candle {
                    timestamp,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
                })
            })
            .collect();

        // Bybit returns newest first; reverse to chronological order
        candles.reverse();
        Ok(candles)
    }

    /// Fetch klines with pagination for longer history.
    pub async fn fetch_klines_extended(
        &self,
        symbol: &str,
        interval: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<Candle>, Box<dyn std::error::Error>> {
        let url = format!("{}/v5/market/kline", self.base_url);
        let mut all_candles = Vec::new();
        let mut current_end = end_time.timestamp_millis();
        let start_ts = start_time.timestamp_millis();

        while current_end > start_ts {
            let response = self
                .client
                .get(&url)
                .query(&[
                    ("category", "spot"),
                    ("symbol", symbol),
                    ("interval", interval),
                    ("limit", "1000"),
                    ("end", &current_end.to_string()),
                ])
                .send()
                .await?;

            let data: BybitResponse = response.json().await?;

            if data.ret_code != 0 {
                break;
            }

            if data.result.list.is_empty() {
                break;
            }

            let candles: Vec<Candle> = data
                .result
                .list
                .iter()
                .filter_map(|row| {
                    if row.len() < 6 {
                        return None;
                    }
                    let ts_ms: i64 = row[0].parse().ok()?;
                    let timestamp =
                        DateTime::from_timestamp(ts_ms / 1000, ((ts_ms % 1000) * 1_000_000) as u32)?;
                    Some(Candle {
                        timestamp,
                        open: row[1].parse().ok()?,
                        high: row[2].parse().ok()?,
                        low: row[3].parse().ok()?,
                        close: row[4].parse().ok()?,
                        volume: row[5].parse().ok()?,
                    })
                })
                .collect();

            if candles.is_empty() {
                break;
            }

            // Get oldest timestamp
            let oldest_ts = candles
                .last()
                .map(|c| c.timestamp.timestamp_millis())
                .unwrap_or(start_ts);

            all_candles.extend(candles);

            if oldest_ts <= start_ts {
                break;
            }
            current_end = oldest_ts - 1;
        }

        // Sort by timestamp and filter to requested range
        all_candles.sort_by_key(|c| c.timestamp);
        all_candles.retain(|c| c.timestamp >= start_time && c.timestamp <= end_time);

        Ok(all_candles)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate synthetic candle data for testing without API access.
pub fn generate_synthetic_candles(n: usize, base_price: f64, symbol: &str) -> Vec<Candle> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut candles = Vec::with_capacity(n);
    let mut price = base_price;
    let now = Utc::now();

    // Use symbol hash for reproducibility
    let seed_offset = symbol.bytes().map(|b| b as f64).sum::<f64>() / 1000.0;

    for i in 0..n {
        let ret = rng.gen_range(-0.03..0.03) + seed_offset * 0.0001;
        let new_price = price * (1.0 + ret);
        let high = price.max(new_price) * (1.0 + rng.gen_range(0.0..0.01));
        let low = price.min(new_price) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100.0..10000.0);

        let timestamp = now - chrono::Duration::days((n - i) as i64);

        candles.push(Candle {
            timestamp,
            open: price,
            high,
            low,
            close: new_price,
            volume,
        });

        price = new_price;
    }

    candles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_synthetic_candles() {
        let candles = generate_synthetic_candles(100, 40000.0, "BTCUSDT");
        assert_eq!(candles.len(), 100);

        for candle in &candles {
            assert!(candle.high >= candle.low);
            assert!(candle.high >= candle.open);
            assert!(candle.high >= candle.close);
            assert!(candle.low <= candle.open);
            assert!(candle.low <= candle.close);
            assert!(candle.volume > 0.0);
        }
    }
}
