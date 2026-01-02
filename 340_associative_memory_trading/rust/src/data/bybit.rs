//! Bybit API Client
//!
//! Fetches cryptocurrency market data from Bybit exchange
//! https://bybit-exchange.github.io/docs/v5/intro

use anyhow::{anyhow, Result};
use chrono::{DateTime, TimeZone, Utc};
use reqwest::blocking::Client;
use serde::Deserialize;
use std::time::Duration;

use super::ohlcv::{OHLCVSeries, OHLCV};

const BYBIT_API_URL: &str = "https://api.bybit.com";
const BYBIT_TESTNET_URL: &str = "https://api-testnet.bybit.com";

/// Bybit API client configuration
#[derive(Clone)]
pub struct BybitConfig {
    pub api_key: Option<String>,
    pub api_secret: Option<String>,
    pub testnet: bool,
}

impl Default for BybitConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            api_secret: None,
            testnet: false,
        }
    }
}

impl BybitConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn testnet(mut self) -> Self {
        self.testnet = true;
        self
    }
}

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
struct ApiResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: T,
}

/// Kline (candlestick) data from API
#[derive(Debug, Deserialize)]
struct KlineResult {
    #[allow(dead_code)]
    symbol: String,
    #[allow(dead_code)]
    category: String,
    list: Vec<Vec<String>>,
}

/// Ticker data from API
#[derive(Debug, Deserialize)]
struct TickerResult {
    #[allow(dead_code)]
    category: String,
    list: Vec<TickerItem>,
}

#[derive(Debug, Deserialize)]
struct TickerItem {
    symbol: String,
    #[serde(rename = "lastPrice")]
    last_price: String,
    #[serde(rename = "highPrice24h")]
    high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    low_price_24h: String,
    #[serde(rename = "volume24h")]
    volume_24h: String,
    #[serde(rename = "turnover24h")]
    turnover_24h: String,
    #[serde(rename = "price24hPcnt")]
    #[allow(dead_code)]
    price_change_24h: Option<String>,
}

/// Market ticker information
#[derive(Debug, Clone)]
pub struct Ticker {
    pub symbol: String,
    pub last_price: f64,
    pub high_24h: f64,
    pub low_24h: f64,
    pub volume_24h: f64,
    pub turnover_24h: f64,
}

/// Bybit API client
pub struct BybitClient {
    client: Client,
    config: BybitConfig,
    base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new(config: BybitConfig) -> Self {
        let base_url = if config.testnet {
            BYBIT_TESTNET_URL.to_string()
        } else {
            BYBIT_API_URL.to_string()
        };

        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            config,
            base_url,
        }
    }

    /// Create client with default config (public API only)
    pub fn public() -> Self {
        Self::new(BybitConfig::default())
    }

    /// Fetch kline/candlestick data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Candle interval: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, M, W
    /// * `limit` - Number of candles (max 1000)
    /// * `start_time` - Optional start timestamp
    /// * `end_time` - Optional end timestamp
    pub fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
        start_time: Option<DateTime<Utc>>,
        end_time: Option<DateTime<Utc>>,
    ) -> Result<OHLCVSeries> {
        let mut params = vec![
            ("category".to_string(), "linear".to_string()),
            ("symbol".to_string(), symbol.to_string()),
            ("interval".to_string(), interval.to_string()),
            ("limit".to_string(), limit.min(1000).to_string()),
        ];

        if let Some(start) = start_time {
            params.push(("start".to_string(), (start.timestamp_millis()).to_string()));
        }

        if let Some(end) = end_time {
            params.push(("end".to_string(), (end.timestamp_millis()).to_string()));
        }

        let query: String = params
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join("&");

        let url = format!("{}/v5/market/kline?{}", self.base_url, query);
        let response = self.client.get(&url).send()?;

        if !response.status().is_success() {
            return Err(anyhow!("API request failed: {}", response.status()));
        }

        let api_response: ApiResponse<KlineResult> = response.json()?;

        if api_response.ret_code != 0 {
            return Err(anyhow!("Bybit API error: {}", api_response.ret_msg));
        }

        // Parse kline data
        // Format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
        let mut data: Vec<OHLCV> = api_response
            .result
            .list
            .iter()
            .filter_map(|kline| {
                if kline.len() >= 6 {
                    let timestamp_ms: i64 = kline[0].parse().ok()?;
                    let timestamp = Utc.timestamp_millis_opt(timestamp_ms).single()?;
                    let open: f64 = kline[1].parse().ok()?;
                    let high: f64 = kline[2].parse().ok()?;
                    let low: f64 = kline[3].parse().ok()?;
                    let close: f64 = kline[4].parse().ok()?;
                    let volume: f64 = kline[5].parse().ok()?;

                    let mut candle = OHLCV::new(timestamp, open, high, low, close, volume);

                    if kline.len() >= 7 {
                        candle.turnover = kline[6].parse().ok();
                    }

                    Some(candle)
                } else {
                    None
                }
            })
            .collect();

        // API returns data in descending order, reverse to ascending
        data.reverse();

        Ok(OHLCVSeries::with_data(
            symbol.to_string(),
            interval.to_string(),
            data,
        ))
    }

    /// Fetch historical klines with pagination
    pub fn get_historical_klines(
        &self,
        symbol: &str,
        interval: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<OHLCVSeries> {
        let mut all_data = Vec::new();
        let mut current_end = end_time;

        log::info!(
            "Fetching historical data for {} from {} to {}",
            symbol,
            start_time,
            end_time
        );

        loop {
            let batch = self.get_klines(symbol, interval, 1000, Some(start_time), Some(current_end))?;

            if batch.is_empty() {
                break;
            }

            let earliest = batch.data.first().unwrap().timestamp;
            log::debug!("Fetched {} candles, earliest: {}", batch.len(), earliest);

            all_data.extend(batch.data);

            if earliest <= start_time || all_data.len() >= 50000 {
                break;
            }

            // Move end time to before the earliest candle
            current_end = earliest - chrono::Duration::milliseconds(1);

            // Rate limiting
            std::thread::sleep(Duration::from_millis(100));
        }

        // Sort by time and remove duplicates
        all_data.sort_by_key(|c| c.timestamp);
        all_data.dedup_by_key(|c| c.timestamp);

        // Filter to requested range
        all_data.retain(|c| c.timestamp >= start_time && c.timestamp <= end_time);

        log::info!("Total candles fetched: {}", all_data.len());

        Ok(OHLCVSeries::with_data(
            symbol.to_string(),
            interval.to_string(),
            all_data,
        ))
    }

    /// Get current ticker information
    pub fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        let url = format!(
            "{}/v5/market/tickers?category=linear&symbol={}",
            self.base_url, symbol
        );

        let response = self.client.get(&url).send()?;

        if !response.status().is_success() {
            return Err(anyhow!("API request failed: {}", response.status()));
        }

        let api_response: ApiResponse<TickerResult> = response.json()?;

        if api_response.ret_code != 0 {
            return Err(anyhow!("Bybit API error: {}", api_response.ret_msg));
        }

        let item = api_response
            .result
            .list
            .first()
            .ok_or_else(|| anyhow!("No ticker data found"))?;

        Ok(Ticker {
            symbol: item.symbol.clone(),
            last_price: item.last_price.parse()?,
            high_24h: item.high_price_24h.parse()?,
            low_24h: item.low_price_24h.parse()?,
            volume_24h: item.volume_24h.parse()?,
            turnover_24h: item.turnover_24h.parse()?,
        })
    }

    /// Get multiple tickers
    pub fn get_tickers(&self, symbols: &[&str]) -> Result<Vec<Ticker>> {
        symbols.iter().map(|s| self.get_ticker(s)).collect()
    }
}

/// Common trading pairs on Bybit
pub mod symbols {
    pub const BTCUSDT: &str = "BTCUSDT";
    pub const ETHUSDT: &str = "ETHUSDT";
    pub const SOLUSDT: &str = "SOLUSDT";
    pub const XRPUSDT: &str = "XRPUSDT";
    pub const DOGEUSDT: &str = "DOGEUSDT";
    pub const AVAXUSDT: &str = "AVAXUSDT";
    pub const ADAUSDT: &str = "ADAUSDT";
    pub const DOTUSDT: &str = "DOTUSDT";
    pub const LINKUSDT: &str = "LINKUSDT";
    pub const MATICUSDT: &str = "MATICUSDT";

    /// Get list of all major trading pairs
    pub fn major_pairs() -> Vec<&'static str> {
        vec![
            BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, DOGEUSDT, AVAXUSDT, ADAUSDT, DOTUSDT, LINKUSDT,
            MATICUSDT,
        ]
    }
}

/// Common intervals
pub mod intervals {
    pub const M1: &str = "1";
    pub const M3: &str = "3";
    pub const M5: &str = "5";
    pub const M15: &str = "15";
    pub const M30: &str = "30";
    pub const H1: &str = "60";
    pub const H2: &str = "120";
    pub const H4: &str = "240";
    pub const H6: &str = "360";
    pub const H12: &str = "720";
    pub const D1: &str = "D";
    pub const W1: &str = "W";
    pub const MN1: &str = "M";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = BybitClient::public();
        assert!(!client.config.testnet);
    }

    #[test]
    fn test_config_builder() {
        let config = BybitConfig::new().testnet();
        assert!(config.testnet);
    }
}
