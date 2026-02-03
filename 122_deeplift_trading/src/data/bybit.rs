//! Bybit API client for fetching cryptocurrency data.

use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Kline (candlestick) data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub turnover: f64,
}

/// Bybit API response.
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
#[derive(Debug, Clone)]
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

    /// Create a client with custom base URL.
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Fetch historical klines from Bybit.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Kline interval ("1", "5", "15", "60", "D")
    /// * `limit` - Number of klines (max 1000)
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
                ("category", "spot"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?;

        let data: BybitResponse = response.json().await?;

        if data.ret_code != 0 {
            anyhow::bail!("API error: {}", data.ret_msg);
        }

        let mut klines: Vec<Kline> = data
            .result
            .list
            .into_iter()
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

        // Bybit returns descending order, reverse to ascending
        klines.reverse();

        info!("Fetched {} klines for {}", klines.len(), symbol);
        Ok(klines)
    }

    /// Fetch klines for multiple symbols.
    pub async fn fetch_multi_symbol(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: usize,
    ) -> Vec<(String, Result<Vec<Kline>>)> {
        let mut results = Vec::new();

        for symbol in symbols {
            let result = self.fetch_klines(symbol, interval, limit).await;
            if let Err(ref e) = result {
                warn!("Failed to fetch {}: {}", symbol, e);
            }
            results.push((symbol.to_string(), result));
        }

        results
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Simulated data generator for testing.
pub struct SimulatedDataGenerator;

impl SimulatedDataGenerator {
    /// Generate random walk klines.
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
            let open_price = price;
            let close_price = price * (1.0 + return_pct);
            let high_price = open_price.max(close_price) * (1.0 + rng.gen::<f64>() * 0.01);
            let low_price = open_price.min(close_price) * (1.0 - rng.gen::<f64>() * 0.01);
            let volume = rng.gen::<f64>() * 1_000_000.0;

            klines.push(Kline {
                timestamp: base_timestamp + i as i64 * 3600000,
                open: open_price,
                high: high_price,
                low: low_price,
                close: close_price,
                volume,
                turnover: volume * close_price,
            });

            price = close_price;
        }

        klines
    }

    /// Generate trending klines.
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
            let open_price = price;
            let close_price = price * (1.0 + return_pct);
            let high_price = open_price.max(close_price) * (1.0 + rng.gen::<f64>() * 0.01);
            let low_price = open_price.min(close_price) * (1.0 - rng.gen::<f64>() * 0.01);
            let volume = rng.gen::<f64>() * 1_000_000.0;

            klines.push(Kline {
                timestamp: base_timestamp + i as i64 * 3600000,
                open: open_price,
                high: high_price,
                low: low_price,
                close: close_price,
                volume,
                turnover: volume * close_price,
            });

            price = close_price;
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
        assert!(klines[0].close > 0.0);
    }

    #[test]
    fn test_trending_data() {
        let klines = SimulatedDataGenerator::generate_trending_klines(100, 50000.0, 0.02, 0.001);
        assert_eq!(klines.len(), 100);
        // With positive trend, final price should generally be higher
        // (not deterministic due to randomness)
    }
}
