//! Bybit API client for fetching cryptocurrency data.
//!
//! This module provides async methods to fetch historical kline (candlestick)
//! data from the Bybit exchange API.

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use crate::MAMLError;

/// A single kline (candlestick) data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Timestamp in milliseconds
    pub timestamp: i64,
    /// Opening price
    pub open: f64,
    /// Highest price
    pub high: f64,
    /// Lowest price
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Trading volume
    pub volume: f64,
    /// Turnover (quote volume)
    pub turnover: f64,
}

impl Kline {
    /// Convert timestamp to DateTime
    pub fn datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(self.timestamp)
            .unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap())
    }
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
#[allow(dead_code)]
struct BybitResult {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

/// Bybit API client
pub struct BybitClient {
    client: reqwest::Client,
    base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Create a client with a custom base URL (for testing)
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Fetch historical klines from Bybit
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Kline interval (e.g., "1", "5", "15", "60", "D")
    /// * `limit` - Number of klines to fetch (max 1000)
    ///
    /// # Returns
    /// Vector of Kline data, sorted by timestamp ascending
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>, MAMLError> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let response = self.client
            .get(&url)
            .query(&[
                ("category", "spot"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await
            .map_err(|e| MAMLError::ApiError(format!("Request failed: {}", e)))?;

        let data: BybitResponse = response
            .json()
            .await
            .map_err(|e| MAMLError::ApiError(format!("Failed to parse response: {}", e)))?;

        if data.ret_code != 0 {
            return Err(MAMLError::ApiError(format!(
                "API error: {} - {}",
                data.ret_code, data.ret_msg
            )));
        }

        let mut klines: Vec<Kline> = data.result.list
            .iter()
            .filter_map(|item| {
                if item.len() >= 7 {
                    Some(Kline {
                        timestamp: item[0].parse().ok()?,
                        open: item[1].parse().ok()?,
                        high: item[2].parse().ok()?,
                        low: item[3].parse().ok()?,
                        close: item[4].parse().ok()?,
                        volume: item[5].parse().ok()?,
                        turnover: item[6].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns data in descending order, so reverse it
        klines.reverse();

        Ok(klines)
    }

    /// Fetch klines for multiple symbols
    pub async fn fetch_multi_symbol_klines(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: usize,
    ) -> Result<Vec<(String, Vec<Kline>)>, MAMLError> {
        let mut results = Vec::new();

        for symbol in symbols {
            let klines = self.fetch_klines(symbol, interval, limit).await?;
            results.push((symbol.to_string(), klines));
        }

        Ok(results)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Simulated data generator for testing without API access
pub struct SimulatedDataGenerator;

impl SimulatedDataGenerator {
    /// Generate simulated kline data
    pub fn generate_klines(num_klines: usize, base_price: f64, volatility: f64) -> Vec<Kline> {
        use rand::Rng;
        use rand_distr::{Distribution, Normal};

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, volatility).unwrap();
        let mut klines = Vec::with_capacity(num_klines);
        let mut price = base_price;
        let base_timestamp = chrono::Utc::now().timestamp_millis() - (num_klines as i64 * 3600000);

        for i in 0..num_klines {
            let return_pct = normal.sample(&mut rng);
            let open = price;
            let close = price * (1.0 + return_pct);
            let high = open.max(close) * (1.0 + rng.gen::<f64>() * 0.01);
            let low = open.min(close) * (1.0 - rng.gen::<f64>() * 0.01);
            let volume = rng.gen::<f64>() * 1000000.0;

            klines.push(Kline {
                timestamp: base_timestamp + (i as i64 * 3600000),
                open,
                high,
                low,
                close,
                volume,
                turnover: volume * close,
            });

            price = close;
        }

        klines
    }

    /// Generate klines with a trend component
    pub fn generate_trending_klines(
        num_klines: usize,
        base_price: f64,
        volatility: f64,
        trend: f64,
    ) -> Vec<Kline> {
        use rand::Rng;
        use rand_distr::{Distribution, Normal};

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, volatility).unwrap();
        let mut klines = Vec::with_capacity(num_klines);
        let mut price = base_price;
        let base_timestamp = chrono::Utc::now().timestamp_millis() - (num_klines as i64 * 3600000);

        for i in 0..num_klines {
            let return_pct = normal.sample(&mut rng) + trend;
            let open = price;
            let close = price * (1.0 + return_pct);
            let high = open.max(close) * (1.0 + rng.gen::<f64>() * 0.01);
            let low = open.min(close) * (1.0 - rng.gen::<f64>() * 0.01);
            let volume = rng.gen::<f64>() * 1000000.0;

            klines.push(Kline {
                timestamp: base_timestamp + (i as i64 * 3600000),
                open,
                high,
                low,
                close,
                volume,
                turnover: volume * close,
            });

            price = close;
        }

        klines
    }

    /// Generate klines with changing market regimes
    pub fn generate_regime_changing_klines(num_klines: usize, base_price: f64) -> Vec<Kline> {
        use rand::Rng;
        use rand_distr::{Distribution, Normal};

        let mut rng = rand::thread_rng();
        let mut klines = Vec::with_capacity(num_klines);
        let mut price = base_price;
        let base_timestamp = chrono::Utc::now().timestamp_millis() - (num_klines as i64 * 3600000);

        // Define regime parameters: (volatility, trend, duration_fraction)
        let regimes = [
            (0.015, 0.0002, 0.2),   // Bull market
            (0.025, -0.0003, 0.15), // Correction
            (0.01, 0.0, 0.15),      // Consolidation
            (0.02, 0.00015, 0.2),   // Recovery
            (0.03, -0.0001, 0.15),  // Volatility spike
            (0.012, 0.0001, 0.15),  // Calm growth
        ];

        let mut current_idx = 0;
        for &(vol, trend, duration_frac) in regimes.iter() {
            let regime_len = (num_klines as f64 * duration_frac) as usize;
            let normal = Normal::new(0.0, vol).unwrap();

            for _ in 0..regime_len {
                if current_idx >= num_klines {
                    break;
                }

                let return_pct = normal.sample(&mut rng) + trend;
                let open = price;
                let close = price * (1.0 + return_pct);
                let high = open.max(close) * (1.0 + rng.gen::<f64>() * 0.01);
                let low = open.min(close) * (1.0 - rng.gen::<f64>() * 0.01);
                let volume = rng.gen::<f64>() * 1000000.0 * (1.0 + vol * 10.0);

                klines.push(Kline {
                    timestamp: base_timestamp + (current_idx as i64 * 3600000),
                    open,
                    high,
                    low,
                    close,
                    volume,
                    turnover: volume * close,
                });

                price = close;
                current_idx += 1;
            }

            if current_idx >= num_klines {
                break;
            }
        }

        // Fill remaining with neutral regime if needed
        let normal = Normal::new(0.0, 0.015).unwrap();
        while klines.len() < num_klines {
            let return_pct = normal.sample(&mut rng);
            let open = price;
            let close = price * (1.0 + return_pct);
            let high = open.max(close) * (1.0 + rng.gen::<f64>() * 0.01);
            let low = open.min(close) * (1.0 - rng.gen::<f64>() * 0.01);
            let volume = rng.gen::<f64>() * 1000000.0;

            klines.push(Kline {
                timestamp: base_timestamp + (klines.len() as i64 * 3600000),
                open,
                high,
                low,
                close,
                volume,
                turnover: volume * close,
            });

            price = close;
        }

        klines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulated_data() {
        let klines = SimulatedDataGenerator::generate_klines(100, 50000.0, 0.02);
        assert_eq!(klines.len(), 100);

        for kline in &klines {
            assert!(kline.high >= kline.low);
            assert!(kline.high >= kline.open.max(kline.close));
            assert!(kline.low <= kline.open.min(kline.close));
            assert!(kline.volume > 0.0);
        }
    }

    #[test]
    fn test_kline_datetime() {
        let kline = Kline {
            timestamp: 1700000000000,
            open: 100.0,
            high: 110.0,
            low: 90.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        let dt = kline.datetime();
        assert!(dt.timestamp() > 0);
    }
}
