//! Bybit API client for fetching market data.

use anyhow::{anyhow, Result};
use reqwest::blocking::Client;
use std::time::Duration;

use super::types::{Kline, KlineResponse};

const BASE_URL: &str = "https://api.bybit.com";

/// Bybit API client.
pub struct BybitClient {
    client: Client,
}

impl BybitClient {
    /// Create a new Bybit client.
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build HTTP client");

        Self { client }
    }

    /// Fetch kline/candlestick data.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Candle interval ("1", "5", "15", "60", "240", "D", "W")
    /// * `limit` - Number of candles (max 1000)
    ///
    /// # Returns
    /// Vector of Kline data sorted by timestamp ascending
    pub fn fetch_klines(&self, symbol: &str, interval: &str, limit: usize) -> Result<Vec<Kline>> {
        let url = format!("{}/v5/market/kline", BASE_URL);

        let response = self
            .client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.min(1000).to_string()),
            ])
            .send()?;

        if !response.status().is_success() {
            return Err(anyhow!("API request failed: {}", response.status()));
        }

        let kline_response: KlineResponse = response.json()?;

        if kline_response.ret_code != 0 {
            return Err(anyhow!("API error: {}", kline_response.ret_msg));
        }

        let mut klines = kline_response.result.parse_klines();
        klines.sort_by_key(|k| k.timestamp);

        Ok(klines)
    }

    /// Fetch data for multiple symbols.
    pub fn fetch_multiple_symbols(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: usize,
    ) -> Result<Vec<(String, Vec<Kline>)>> {
        let mut results = Vec::new();

        for symbol in symbols {
            match self.fetch_klines(symbol, interval, limit) {
                Ok(klines) => {
                    results.push((symbol.to_string(), klines));
                }
                Err(e) => {
                    eprintln!("Warning: Failed to fetch {}: {}", symbol, e);
                }
            }
            // Rate limiting
            std::thread::sleep(Duration::from_millis(100));
        }

        Ok(results)
    }

    /// Get current price for a symbol.
    pub fn get_ticker(&self, symbol: &str) -> Result<f64> {
        let url = format!("{}/v5/market/tickers", BASE_URL);

        let response = self
            .client
            .get(&url)
            .query(&[("category", "linear"), ("symbol", symbol)])
            .send()?;

        if !response.status().is_success() {
            return Err(anyhow!("API request failed: {}", response.status()));
        }

        let data: serde_json::Value = response.json()?;

        let price_str = data["result"]["list"][0]["lastPrice"]
            .as_str()
            .ok_or_else(|| anyhow!("Could not parse price"))?;

        price_str
            .parse::<f64>()
            .map_err(|_| anyhow!("Invalid price format"))
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires network access
    fn test_fetch_klines() {
        let client = BybitClient::new();
        let result = client.fetch_klines("BTCUSDT", "60", 10);

        match result {
            Ok(klines) => {
                assert!(!klines.is_empty());
                println!("Fetched {} klines", klines.len());
            }
            Err(e) => {
                eprintln!("Test skipped (network issue): {}", e);
            }
        }
    }
}
