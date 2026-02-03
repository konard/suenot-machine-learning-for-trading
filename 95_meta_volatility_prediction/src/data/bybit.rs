//! Bybit API client for fetching cryptocurrency market data.

use serde::Deserialize;
use tracing::info;

/// Bybit API client for market data.
pub struct BybitClient {
    base_url: String,
    client: reqwest::Client,
}

#[derive(Debug, Deserialize)]
struct BybitResponse {
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// OHLCV candle data.
#[derive(Debug, Clone)]
pub struct Kline {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub turnover: f64,
}

impl BybitClient {
    /// Create a new Bybit client.
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Fetch kline (candlestick) data for a symbol.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Kline interval in minutes (e.g., "60" for 1h)
    /// * `limit` - Number of candles to fetch (max 1000)
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Kline>> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let resp: BybitResponse = self
            .client
            .get(&url)
            .query(&[
                ("category", "spot"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?
            .json()
            .await?;

        let mut klines: Vec<Kline> = resp
            .result
            .list
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 7 {
                    Some(Kline {
                        timestamp: row[0].parse().unwrap_or(0),
                        open: row[1].parse().unwrap_or(0.0),
                        high: row[2].parse().unwrap_or(0.0),
                        low: row[3].parse().unwrap_or(0.0),
                        close: row[4].parse().unwrap_or(0.0),
                        volume: row[5].parse().unwrap_or(0.0),
                        turnover: row[6].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first, reverse to chronological order
        klines.reverse();

        info!(
            symbol = symbol,
            count = klines.len(),
            "Fetched klines from Bybit"
        );

        Ok(klines)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute log returns from klines.
pub fn compute_log_returns(klines: &[Kline]) -> Vec<f64> {
    klines
        .windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_log_returns() {
        let klines = vec![
            Kline { timestamp: 0, open: 100.0, high: 105.0, low: 99.0, close: 100.0, volume: 1000.0, turnover: 100000.0 },
            Kline { timestamp: 1, open: 100.0, high: 110.0, low: 100.0, close: 105.0, volume: 1200.0, turnover: 126000.0 },
            Kline { timestamp: 2, open: 105.0, high: 106.0, low: 102.0, close: 103.0, volume: 900.0, turnover: 92700.0 },
        ];

        let returns = compute_log_returns(&klines);
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - (105.0_f64 / 100.0).ln()).abs() < 1e-10);
        assert!((returns[1] - (103.0_f64 / 105.0).ln()).abs() < 1e-10);
    }
}
