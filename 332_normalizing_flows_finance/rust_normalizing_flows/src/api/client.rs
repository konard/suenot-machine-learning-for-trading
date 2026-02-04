//! Bybit API client for fetching market data.

use crate::api::types::{parse_klines, ApiResponse, Kline, KlineResult};
use anyhow::Result;
use reqwest::blocking::Client;
use std::collections::HashMap;

/// Bybit REST API client
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
    /// Create a new Bybit client
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Create client for testnet
    pub fn testnet() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api-testnet.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data synchronously
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Timeframe ("1", "5", "15", "60", "240", "D", "W")
    /// * `limit` - Number of candles to fetch (max 1000)
    ///
    /// # Returns
    /// Vector of Kline data sorted by timestamp (oldest first)
    pub fn get_klines_sync(&self, symbol: &str, interval: &str, limit: usize) -> Result<Vec<Kline>> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let mut params = HashMap::new();
        params.insert("category", "spot");
        params.insert("symbol", symbol);
        params.insert("interval", interval);

        let limit_str = limit.to_string();
        params.insert("limit", &limit_str);

        let response: ApiResponse<KlineResult> = self.client.get(&url).query(&params).send()?.json()?;

        if response.ret_code != 0 {
            anyhow::bail!("API error: {}", response.ret_msg);
        }

        let mut klines = parse_klines(&response.result);

        // Sort by timestamp (API returns newest first)
        klines.sort_by_key(|k| k.timestamp);

        Ok(klines)
    }

    /// Fetch ticker data for a symbol
    pub fn get_ticker_sync(&self, symbol: &str) -> Result<serde_json::Value> {
        let url = format!("{}/v5/market/tickers", self.base_url);

        let mut params = HashMap::new();
        params.insert("category", "spot");
        params.insert("symbol", symbol);

        let response: serde_json::Value = self.client.get(&url).query(&params).send()?.json()?;

        Ok(response)
    }

    /// Fetch multiple symbols' klines
    pub fn get_multiple_klines_sync(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: usize,
    ) -> Result<HashMap<String, Vec<Kline>>> {
        let mut result = HashMap::new();

        for symbol in symbols {
            match self.get_klines_sync(symbol, interval, limit) {
                Ok(klines) => {
                    result.insert(symbol.to_string(), klines);
                }
                Err(e) => {
                    log::warn!("Failed to fetch {}: {}", symbol, e);
                }
            }
            // Small delay to respect rate limits
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = BybitClient::new();
        assert_eq!(client.base_url, "https://api.bybit.com");
    }
}
