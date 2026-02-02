//! Bybit API client for fetching cryptocurrency kline data.

use serde::Deserialize;
use chrono::{DateTime, Utc, TimeZone};

/// A single kline (candlestick) from Bybit.
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

impl Kline {
    /// Convert timestamp to DateTime.
    pub fn datetime(&self) -> DateTime<Utc> {
        Utc.timestamp_millis_opt(self.timestamp).unwrap()
    }

    /// Compute the log return relative to a previous kline.
    pub fn log_return(&self, prev: &Kline) -> f64 {
        (self.close / prev.close).ln()
    }

    /// Compute simple return relative to a previous kline.
    pub fn simple_return(&self, prev: &Kline) -> f64 {
        self.close / prev.close - 1.0
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

/// Async client for the Bybit API.
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

    /// Fetch kline data for a symbol.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT").
    /// * `interval` - Kline interval ("1", "5", "15", "60", "240", "D").
    /// * `limit` - Number of klines (max 1000).
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Kline>> {
        let url = format!("{}/v5/market/kline", self.base_url);
        let resp = self
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

        let body: BybitResponse = resp.json().await?;
        if body.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", body.ret_msg);
        }

        let mut klines: Vec<Kline> = body
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() < 7 {
                    return None;
                }
                Some(Kline {
                    timestamp: row[0].parse().ok()?,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
                    turnover: row[6].parse().ok()?,
                })
            })
            .collect();

        // Bybit returns data in descending order; reverse to ascending
        klines.reverse();
        Ok(klines)
    }

    /// Fetch klines for multiple symbols.
    pub async fn fetch_multi_symbol_klines(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<(String, Vec<Kline>)>> {
        let mut results = Vec::new();
        for &symbol in symbols {
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

/// Extract simple returns from a series of klines.
pub fn klines_to_returns(klines: &[Kline]) -> Vec<f64> {
    if klines.len() < 2 {
        return vec![];
    }
    klines
        .windows(2)
        .map(|w| w[1].simple_return(&w[0]))
        .collect()
}
