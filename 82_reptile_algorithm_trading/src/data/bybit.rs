//! Bybit API client for fetching cryptocurrency market data.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use crate::{Result, ReptileError};

/// Kline (candlestick) data from Bybit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Start timestamp
    pub timestamp: DateTime<Utc>,
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
    /// Turnover (volume * price)
    pub turnover: f64,
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
#[derive(Debug, Clone)]
pub struct BybitClient {
    base_url: String,
    client: reqwest::Client,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Create a client with a custom base URL
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Fetch historical klines (candlesticks) from Bybit
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Time interval (e.g., "1", "5", "15", "60", "D")
    /// * `limit` - Number of candles to fetch (max 1000)
    ///
    /// # Returns
    /// Vector of Kline data sorted by timestamp (oldest first)
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>> {
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
            .map_err(|e| ReptileError::ApiError(e.to_string()))?;

        let data: BybitResponse = response
            .json()
            .await
            .map_err(|e| ReptileError::ApiError(e.to_string()))?;

        if data.ret_code != 0 {
            return Err(ReptileError::ApiError(data.ret_msg));
        }

        let mut klines: Vec<Kline> = data.result.list
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 7 {
                    let timestamp_ms: i64 = row[0].parse().ok()?;
                    Some(Kline {
                        timestamp: DateTime::from_timestamp_millis(timestamp_ms)?,
                        open: row[1].parse().ok()?,
                        high: row[2].parse().ok()?,
                        low: row[3].parse().ok()?,
                        close: row[4].parse().ok()?,
                        volume: row[5].parse().ok()?,
                        turnover: row[6].parse().ok()?,
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

    /// Get current price for a symbol
    pub async fn get_ticker(&self, symbol: &str) -> Result<f64> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        let response = self.client
            .get(&url)
            .query(&[
                ("category", "spot"),
                ("symbol", symbol),
            ])
            .send()
            .await
            .map_err(|e| ReptileError::ApiError(e.to_string()))?;

        let text = response.text().await
            .map_err(|e| ReptileError::ApiError(e.to_string()))?;

        // Parse the last price from the response
        let data: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| ReptileError::ApiError(e.to_string()))?;

        data["result"]["list"][0]["lastPrice"]
            .as_str()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| ReptileError::ApiError("Failed to parse ticker price".to_string()))
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Simulated market data for testing without API access
#[derive(Debug)]
pub struct SimulatedDataSource {
    #[allow(dead_code)]
    symbol: String,
    initial_price: f64,
    volatility: f64,
}

impl SimulatedDataSource {
    /// Create a new simulated data source
    pub fn new(symbol: &str, initial_price: f64, volatility: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            initial_price,
            volatility,
        }
    }

    /// Generate simulated klines
    pub fn generate_klines(&self, count: usize) -> Vec<Kline> {
        use rand::Rng;
        use rand_distr::{Distribution, Normal};

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, self.volatility).unwrap();

        let mut klines = Vec::with_capacity(count);
        let mut price = self.initial_price;
        let mut timestamp = Utc::now() - chrono::Duration::hours(count as i64);

        for _ in 0..count {
            let change = normal.sample(&mut rng);
            let open = price;
            let close = price * (1.0 + change);

            let high = open.max(close) * (1.0 + rng.gen::<f64>() * self.volatility.abs());
            let low = open.min(close) * (1.0 - rng.gen::<f64>() * self.volatility.abs());

            let volume = rng.gen_range(100.0..10000.0);
            let turnover = volume * (open + close) / 2.0;

            klines.push(Kline {
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                turnover,
            });

            price = close;
            timestamp = timestamp + chrono::Duration::hours(1);
        }

        klines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulated_data() {
        let source = SimulatedDataSource::new("BTCUSDT", 50000.0, 0.02);
        let klines = source.generate_klines(100);

        assert_eq!(klines.len(), 100);
        for kline in &klines {
            assert!(kline.high >= kline.low);
            assert!(kline.high >= kline.open);
            assert!(kline.high >= kline.close);
            assert!(kline.low <= kline.open);
            assert!(kline.low <= kline.close);
        }
    }

    #[test]
    fn test_client_creation() {
        let client = BybitClient::new();
        assert!(client.base_url.contains("bybit.com"));
    }
}
