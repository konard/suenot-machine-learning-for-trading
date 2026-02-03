//! Bybit API client for fetching cryptocurrency market data.

use crate::{Result, ShapError};
use chrono::{DateTime, Utc};
use rand::prelude::*;
use rand_distr::LogNormal;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// OHLCV candle data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Candle {
    /// Calculate the return from open to close.
    pub fn return_pct(&self) -> f64 {
        (self.close - self.open) / self.open
    }

    /// Calculate the range (high - low).
    pub fn range(&self) -> f64 {
        self.high - self.low
    }
}

/// Market data container.
#[derive(Debug, Clone)]
pub struct MarketData {
    pub candles: Vec<Candle>,
    pub symbol: String,
    pub interval: String,
    pub source: String,
}

impl MarketData {
    /// Get close prices as a vector.
    pub fn close_prices(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.close).collect()
    }

    /// Get returns as a vector.
    pub fn returns(&self) -> Vec<f64> {
        let prices = self.close_prices();
        let mut returns = vec![0.0];
        for i in 1..prices.len() {
            returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
        }
        returns
    }

    /// Get volumes as a vector.
    pub fn volumes(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.volume).collect()
    }
}

/// Bybit API response structure.
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

/// Bybit API client.
pub struct BybitClient {
    client: reqwest::Client,
    base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (OHLCV) data from Bybit.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair (e.g., "BTCUSDT").
    /// * `interval` - Kline interval (e.g., "60" for 1 hour).
    /// * `limit` - Number of candles to fetch (max 1000).
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<MarketData> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let params = [
            ("category", "linear"),
            ("symbol", symbol),
            ("interval", interval),
            ("limit", &limit.min(1000).to_string()),
        ];

        match self.client.get(&url).query(&params).send().await {
            Ok(response) => {
                if !response.status().is_success() {
                    warn!("Bybit API returned error status, using synthetic data");
                    return Ok(self.generate_synthetic(symbol, interval, limit));
                }

                match response.json::<BybitResponse>().await {
                    Ok(data) => {
                        if data.ret_code != 0 {
                            warn!("Bybit API error: {}, using synthetic data", data.ret_msg);
                            return Ok(self.generate_synthetic(symbol, interval, limit));
                        }

                        let candles = self.parse_candles(&data.result.list)?;
                        info!("Fetched {} candles for {} from Bybit", candles.len(), symbol);

                        Ok(MarketData {
                            candles,
                            symbol: symbol.to_string(),
                            interval: interval.to_string(),
                            source: "bybit".to_string(),
                        })
                    }
                    Err(e) => {
                        warn!("Failed to parse Bybit response: {}, using synthetic data", e);
                        Ok(self.generate_synthetic(symbol, interval, limit))
                    }
                }
            }
            Err(e) => {
                warn!("Failed to fetch from Bybit: {}, using synthetic data", e);
                Ok(self.generate_synthetic(symbol, interval, limit))
            }
        }
    }

    /// Parse candles from Bybit API response.
    fn parse_candles(&self, list: &[Vec<String>]) -> Result<Vec<Candle>> {
        let mut candles: Vec<Candle> = list
            .iter()
            .filter_map(|row| {
                if row.len() < 6 {
                    return None;
                }
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
            })
            .collect();

        // Sort by timestamp ascending
        candles.sort_by_key(|c| c.timestamp);
        Ok(candles)
    }

    /// Generate synthetic OHLCV data.
    fn generate_synthetic(&self, symbol: &str, interval: &str, n_points: usize) -> MarketData {
        let mut rng = rand::thread_rng();
        let base_price = if symbol.contains("BTC") { 50000.0 } else { 3000.0 };
        let volatility = 0.02;

        let mut candles = Vec::with_capacity(n_points);
        let mut price = base_price;

        let log_normal = LogNormal::new(10.0, 1.0).unwrap();

        for i in 0..n_points {
            // Generate return with volatility clustering
            let ret: f64 = rng.gen::<f64>() * 2.0 * volatility - volatility;
            let new_price = price * (1.0 + ret);

            let noise = rng.gen::<f64>() * volatility * price;
            let high = new_price + noise.abs();
            let low = new_price - noise.abs();

            let volume: f64 = rng.sample(log_normal);

            let timestamp = Utc::now() - chrono::Duration::hours((n_points - i) as i64);

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

        info!("Generated {} synthetic candles for {}", n_points, symbol);

        MarketData {
            candles,
            symbol: symbol.to_string(),
            interval: interval.to_string(),
            source: "synthetic".to_string(),
        }
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}
