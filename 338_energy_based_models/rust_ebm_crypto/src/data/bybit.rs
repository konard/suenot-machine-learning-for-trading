//! Bybit API client for fetching market data

use anyhow::{anyhow, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::ohlcv::{Candle, OhlcvData};

/// Bybit API configuration
#[derive(Debug, Clone)]
pub struct BybitConfig {
    /// Base URL for API
    pub base_url: String,
    /// API key (optional, for private endpoints)
    pub api_key: Option<String>,
    /// API secret (optional, for private endpoints)
    pub api_secret: Option<String>,
    /// Request timeout in seconds
    pub timeout: u64,
}

impl Default for BybitConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            api_key: None,
            api_secret: None,
            timeout: 30,
        }
    }
}

impl BybitConfig {
    /// Create config for testnet
    pub fn testnet() -> Self {
        Self {
            base_url: "https://api-testnet.bybit.com".to_string(),
            ..Default::default()
        }
    }
}

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BybitResponse<T> {
    ret_code: i32,
    ret_msg: String,
    result: T,
}

/// Kline response from Bybit
#[derive(Debug, Deserialize)]
struct KlineResult {
    category: String,
    symbol: String,
    list: Vec<Vec<String>>,
}

/// Bybit API client
#[derive(Debug, Clone)]
pub struct BybitClient {
    config: BybitConfig,
    client: Client,
}

impl BybitClient {
    /// Create a new Bybit client with custom config
    pub fn new(config: BybitConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout))
            .build()?;

        Ok(Self { config, client })
    }

    /// Create a client for public endpoints only
    pub fn public() -> Self {
        Self::new(BybitConfig::default()).expect("Failed to create client")
    }

    /// Create a client for testnet
    pub fn testnet() -> Self {
        Self::new(BybitConfig::testnet()).expect("Failed to create client")
    }

    /// Fetch kline (candlestick) data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Kline interval: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
    /// * `limit` - Number of candles to fetch (max 1000)
    /// * `start` - Start timestamp in milliseconds (optional)
    /// * `end` - End timestamp in milliseconds (optional)
    pub fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
        start: Option<i64>,
        end: Option<i64>,
    ) -> Result<OhlcvData> {
        let limit = limit.min(1000);

        let mut url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.config.base_url, symbol, interval, limit
        );

        if let Some(start_ts) = start {
            url.push_str(&format!("&start={}", start_ts));
        }
        if let Some(end_ts) = end {
            url.push_str(&format!("&end={}", end_ts));
        }

        log::debug!("Fetching klines from: {}", url);

        let response = self.client.get(&url).send()?;
        let text = response.text()?;

        let parsed: BybitResponse<KlineResult> = serde_json::from_str(&text)
            .map_err(|e| anyhow!("Failed to parse response: {} - Response: {}", e, text))?;

        if parsed.ret_code != 0 {
            return Err(anyhow!(
                "Bybit API error {}: {}",
                parsed.ret_code,
                parsed.ret_msg
            ));
        }

        let mut ohlcv = OhlcvData::new(symbol, interval);

        for item in parsed.result.list {
            if item.len() >= 6 {
                let candle = Candle {
                    timestamp: item[0].parse().unwrap_or(0),
                    open: item[1].parse().unwrap_or(0.0),
                    high: item[2].parse().unwrap_or(0.0),
                    low: item[3].parse().unwrap_or(0.0),
                    close: item[4].parse().unwrap_or(0.0),
                    volume: item[5].parse().unwrap_or(0.0),
                    turnover: item.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                };
                ohlcv.push(candle);
            }
        }

        // Bybit returns data in reverse order (newest first), so we reverse
        ohlcv.data.reverse();

        Ok(ohlcv)
    }

    /// Fetch historical klines with pagination
    ///
    /// Fetches more than 1000 candles by making multiple requests
    pub fn get_historical_klines(
        &self,
        symbol: &str,
        interval: &str,
        total_candles: usize,
        end_time: Option<i64>,
    ) -> Result<OhlcvData> {
        let mut all_data = OhlcvData::new(symbol, interval);
        let mut current_end = end_time;
        let mut remaining = total_candles;

        let progress = indicatif::ProgressBar::new(total_candles as u64);
        progress.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} candles")?
                .progress_chars("##-"),
        );

        while remaining > 0 {
            let limit = remaining.min(1000) as u32;

            let batch = self.get_klines(symbol, interval, limit, None, current_end)?;

            if batch.is_empty() {
                break;
            }

            // Get the earliest timestamp for next request
            if let Some(first) = batch.data.first() {
                current_end = Some(first.timestamp - 1);
            }

            remaining = remaining.saturating_sub(batch.len());
            progress.inc(batch.len() as u64);

            // Prepend batch to all_data (since we're going backwards)
            let mut new_data = batch.data;
            new_data.append(&mut all_data.data);
            all_data.data = new_data;

            // Small delay to avoid rate limiting
            std::thread::sleep(Duration::from_millis(100));
        }

        progress.finish_with_message("Done");
        Ok(all_data)
    }

    /// Get available trading symbols
    pub fn get_symbols(&self) -> Result<Vec<SymbolInfo>> {
        let url = format!(
            "{}/v5/market/instruments-info?category=linear",
            self.config.base_url
        );

        let response = self.client.get(&url).send()?;
        let parsed: BybitResponse<InstrumentsResult> = response.json()?;

        if parsed.ret_code != 0 {
            return Err(anyhow!(
                "Bybit API error {}: {}",
                parsed.ret_code,
                parsed.ret_msg
            ));
        }

        Ok(parsed.result.list)
    }

    /// Get ticker information
    pub fn get_ticker(&self, symbol: &str) -> Result<TickerInfo> {
        let url = format!(
            "{}/v5/market/tickers?category=linear&symbol={}",
            self.config.base_url, symbol
        );

        let response = self.client.get(&url).send()?;
        let parsed: BybitResponse<TickerResult> = response.json()?;

        if parsed.ret_code != 0 {
            return Err(anyhow!(
                "Bybit API error {}: {}",
                parsed.ret_code,
                parsed.ret_msg
            ));
        }

        parsed
            .result
            .list
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No ticker data found for {}", symbol))
    }
}

/// Symbol information
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SymbolInfo {
    pub symbol: String,
    pub base_coin: String,
    pub quote_coin: String,
    pub status: String,
}

/// Instruments response
#[derive(Debug, Deserialize)]
struct InstrumentsResult {
    list: Vec<SymbolInfo>,
}

/// Ticker information
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TickerInfo {
    pub symbol: String,
    pub last_price: String,
    pub high_price24h: String,
    pub low_price24h: String,
    pub volume24h: String,
    pub turnover24h: String,
    pub price24h_pcnt: String,
}

/// Ticker response
#[derive(Debug, Deserialize)]
struct TickerResult {
    list: Vec<TickerInfo>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires network
    fn test_fetch_klines() {
        let client = BybitClient::public();
        let data = client.get_klines("BTCUSDT", "60", 10, None, None);
        assert!(data.is_ok());

        let ohlcv = data.unwrap();
        assert!(!ohlcv.is_empty());
        assert!(ohlcv.len() <= 10);
    }
}
