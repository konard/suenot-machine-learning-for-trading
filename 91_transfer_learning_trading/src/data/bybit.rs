//! Bybit API integration for fetching cryptocurrency market data.
//!
//! Supports fetching historical klines (candlesticks), tickers, and order book data.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use crate::data::features::OHLCVBar;

/// Kline interval for Bybit API.
#[derive(Debug, Clone, Copy)]
pub enum KlineInterval {
    Min1,
    Min5,
    Min15,
    Min30,
    Hour1,
    Hour4,
    Day1,
    Week1,
}

impl KlineInterval {
    pub fn as_str(&self) -> &str {
        match self {
            KlineInterval::Min1 => "1",
            KlineInterval::Min5 => "5",
            KlineInterval::Min15 => "15",
            KlineInterval::Min30 => "30",
            KlineInterval::Hour1 => "60",
            KlineInterval::Hour4 => "240",
            KlineInterval::Day1 => "D",
            KlineInterval::Week1 => "W",
        }
    }
}

/// Configuration for Bybit API client.
#[derive(Debug, Clone)]
pub struct BybitConfig {
    /// Base URL for the Bybit API
    pub base_url: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
}

impl Default for BybitConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            timeout_secs: 30,
        }
    }
}

/// Bybit API response wrapper.
#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: T,
}

/// Kline response result.
#[derive(Debug, Deserialize)]
struct KlineResult {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

/// Ticker data from Bybit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickerData {
    pub symbol: String,
    pub last_price: f64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub volume_24h: f64,
    pub high_24h: f64,
    pub low_24h: f64,
    pub turnover_24h: f64,
}

/// Bybit API client for fetching market data.
#[derive(Debug, Clone)]
pub struct BybitClient {
    config: BybitConfig,
    client: reqwest::Client,
}

impl BybitClient {
    /// Create a new Bybit client with the given configuration.
    pub fn new(config: BybitConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self { config, client }
    }

    /// Fetch historical klines (candlestick data) for a symbol.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Kline interval
    /// * `limit` - Number of klines to fetch (max 200)
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: KlineInterval,
        limit: usize,
    ) -> Result<Vec<OHLCVBar>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.config.base_url,
            symbol,
            interval.as_str(),
            limit.min(200),
        );

        let response: BybitResponse<KlineResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {} (code: {})", response.ret_msg, response.ret_code);
        }

        let bars: Vec<OHLCVBar> = response
            .result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() >= 6 {
                    Some(OHLCVBar::new(
                        item[0].parse::<i64>().unwrap_or(0),
                        item[1].parse::<f64>().unwrap_or(0.0),
                        item[2].parse::<f64>().unwrap_or(0.0),
                        item[3].parse::<f64>().unwrap_or(0.0),
                        item[4].parse::<f64>().unwrap_or(0.0),
                        item[5].parse::<f64>().unwrap_or(0.0),
                    ))
                } else {
                    None
                }
            })
            .collect();

        Ok(bars)
    }

    /// Fetch current ticker data for a symbol.
    pub async fn fetch_ticker(&self, symbol: &str) -> Result<TickerData> {
        let url = format!(
            "{}/v5/market/tickers?category=linear&symbol={}",
            self.config.base_url, symbol,
        );

        let response: serde_json::Value = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        let ticker_list = response["result"]["list"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Invalid ticker response"))?;

        let ticker = ticker_list
            .first()
            .ok_or_else(|| anyhow::anyhow!("No ticker data found for {}", symbol))?;

        Ok(TickerData {
            symbol: symbol.to_string(),
            last_price: parse_f64(&ticker["lastPrice"]),
            bid_price: parse_f64(&ticker["bid1Price"]),
            ask_price: parse_f64(&ticker["ask1Price"]),
            volume_24h: parse_f64(&ticker["volume24h"]),
            high_24h: parse_f64(&ticker["highPrice24h"]),
            low_24h: parse_f64(&ticker["lowPrice24h"]),
            turnover_24h: parse_f64(&ticker["turnover24h"]),
        })
    }

    /// Fetch klines for multiple symbols (source domain data).
    pub async fn fetch_multi_klines(
        &self,
        symbols: &[&str],
        interval: KlineInterval,
        limit: usize,
    ) -> Result<Vec<(String, Vec<OHLCVBar>)>> {
        let mut results = Vec::new();
        for symbol in symbols {
            let bars = self.fetch_klines(symbol, interval, limit).await?;
            results.push((symbol.to_string(), bars));
        }
        Ok(results)
    }
}

fn parse_f64(value: &serde_json::Value) -> f64 {
    value
        .as_str()
        .and_then(|s| s.parse::<f64>().ok())
        .or_else(|| value.as_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_interval() {
        assert_eq!(KlineInterval::Hour1.as_str(), "60");
        assert_eq!(KlineInterval::Day1.as_str(), "D");
        assert_eq!(KlineInterval::Min5.as_str(), "5");
    }

    #[test]
    fn test_bybit_config_default() {
        let config = BybitConfig::default();
        assert!(config.base_url.contains("bybit.com"));
        assert_eq!(config.timeout_secs, 30);
    }

    #[test]
    fn test_parse_f64() {
        assert_eq!(parse_f64(&serde_json::json!("123.45")), 123.45);
        assert_eq!(parse_f64(&serde_json::json!(67.89)), 67.89);
        assert_eq!(parse_f64(&serde_json::json!(null)), 0.0);
    }
}
