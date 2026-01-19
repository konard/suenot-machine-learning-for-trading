//! Market Data Module
//!
//! Provides data loaders for various market data sources including
//! stocks (Yahoo Finance) and cryptocurrency (Bybit).

use crate::error::{Error, Result};
use async_trait::async_trait;
use chrono::{DateTime, Duration, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// OHLCV bar data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCVBar {
    /// Bar timestamp
    pub timestamp: DateTime<Utc>,
    /// Open price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Close price
    pub close: f64,
    /// Volume
    pub volume: f64,
}

/// Data source type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataSource {
    /// Yahoo Finance
    Yahoo,
    /// Bybit cryptocurrency exchange
    Bybit,
    /// Mock data for testing
    Mock,
}

/// Interval for data bars
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Interval {
    /// 1 minute
    Min1,
    /// 5 minutes
    Min5,
    /// 15 minutes
    Min15,
    /// 1 hour
    Hour1,
    /// 4 hours
    Hour4,
    /// 1 day
    Day1,
    /// 1 week
    Week1,
}

impl Interval {
    /// Convert to Yahoo Finance interval string
    pub fn to_yahoo_interval(&self) -> &str {
        match self {
            Interval::Min1 => "1m",
            Interval::Min5 => "5m",
            Interval::Min15 => "15m",
            Interval::Hour1 => "1h",
            Interval::Hour4 => "4h",
            Interval::Day1 => "1d",
            Interval::Week1 => "1wk",
        }
    }

    /// Convert to Bybit interval string
    pub fn to_bybit_interval(&self) -> &str {
        match self {
            Interval::Min1 => "1",
            Interval::Min5 => "5",
            Interval::Min15 => "15",
            Interval::Hour1 => "60",
            Interval::Hour4 => "240",
            Interval::Day1 => "D",
            Interval::Week1 => "W",
        }
    }

    /// Get duration in seconds
    pub fn to_seconds(&self) -> i64 {
        match self {
            Interval::Min1 => 60,
            Interval::Min5 => 300,
            Interval::Min15 => 900,
            Interval::Hour1 => 3600,
            Interval::Hour4 => 14400,
            Interval::Day1 => 86400,
            Interval::Week1 => 604800,
        }
    }
}

/// Trait for market data loaders
#[async_trait]
pub trait DataLoader: Send + Sync {
    /// Fetch OHLCV data for a symbol
    async fn fetch_ohlcv(
        &self,
        symbol: &str,
        interval: Interval,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<OHLCVBar>>;

    /// Get the data source name
    fn source(&self) -> DataSource;
}

/// Main market data loader
pub struct MarketDataLoader {
    loader: Box<dyn DataLoader>,
}

impl MarketDataLoader {
    /// Create Yahoo Finance loader
    pub fn yahoo() -> Self {
        Self {
            loader: Box::new(YahooLoader::new()),
        }
    }

    /// Create Bybit loader
    pub fn bybit() -> Self {
        Self {
            loader: Box::new(BybitLoader::new()),
        }
    }

    /// Create mock loader for testing
    pub fn mock() -> Self {
        Self {
            loader: Box::new(MockLoader::new()),
        }
    }

    /// Fetch OHLCV data
    pub async fn fetch_ohlcv(
        &self,
        symbol: &str,
        interval: Interval,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<OHLCVBar>> {
        self.loader.fetch_ohlcv(symbol, interval, start, end).await
    }

    /// Get data source
    pub fn source(&self) -> DataSource {
        self.loader.source()
    }
}

/// Yahoo Finance data loader
pub struct YahooLoader {
    client: reqwest::Client,
}

impl YahooLoader {
    /// Create new Yahoo Finance loader
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

impl Default for YahooLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DataLoader for YahooLoader {
    async fn fetch_ohlcv(
        &self,
        symbol: &str,
        interval: Interval,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<OHLCVBar>> {
        let period1 = start.timestamp();
        let period2 = end.timestamp();

        let url = format!(
            "https://query1.finance.yahoo.com/v8/finance/chart/{}?period1={}&period2={}&interval={}",
            symbol,
            period1,
            period2,
            interval.to_yahoo_interval()
        );

        let response = self.client
            .get(&url)
            .header("User-Agent", "Mozilla/5.0")
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Error::DataSourceError(format!(
                "Yahoo Finance error: {}",
                response.status()
            )));
        }

        let json: serde_json::Value = response.json().await?;

        // Parse Yahoo Finance response
        let chart = json["chart"]["result"][0].clone();
        let timestamps = chart["timestamp"]
            .as_array()
            .ok_or_else(|| Error::ParseError("No timestamps in response".to_string()))?;

        let quote = &chart["indicators"]["quote"][0];
        let opens = quote["open"].as_array();
        let highs = quote["high"].as_array();
        let lows = quote["low"].as_array();
        let closes = quote["close"].as_array();
        let volumes = quote["volume"].as_array();

        let mut bars = Vec::new();

        for i in 0..timestamps.len() {
            let ts = timestamps[i].as_i64().unwrap_or(0);
            let timestamp = DateTime::from_timestamp(ts, 0)
                .unwrap_or(Utc::now());

            let open = opens.and_then(|a| a.get(i)).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let high = highs.and_then(|a| a.get(i)).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let low = lows.and_then(|a| a.get(i)).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let close = closes.and_then(|a| a.get(i)).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let volume = volumes.and_then(|a| a.get(i)).and_then(|v| v.as_f64()).unwrap_or(0.0);

            // Skip bars with missing data
            if open == 0.0 || close == 0.0 {
                continue;
            }

            bars.push(OHLCVBar {
                timestamp,
                open,
                high,
                low,
                close,
                volume,
            });
        }

        Ok(bars)
    }

    fn source(&self) -> DataSource {
        DataSource::Yahoo
    }
}

/// Bybit cryptocurrency exchange data loader
pub struct BybitLoader {
    client: reqwest::Client,
    base_url: String,
}

impl BybitLoader {
    /// Create new Bybit loader
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Create with testnet
    pub fn testnet() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "https://api-testnet.bybit.com".to_string(),
        }
    }
}

impl Default for BybitLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DataLoader for BybitLoader {
    async fn fetch_ohlcv(
        &self,
        symbol: &str,
        interval: Interval,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<OHLCVBar>> {
        let start_ms = start.timestamp_millis();
        let end_ms = end.timestamp_millis();

        // Bybit uses different endpoints for different products
        // We'll use the spot market klines endpoint
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&start={}&end={}&limit=1000",
            self.base_url,
            symbol,
            interval.to_bybit_interval(),
            start_ms,
            end_ms
        );

        let response = self.client
            .get(&url)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(Error::DataSourceError(format!(
                "Bybit API error: {}",
                response.status()
            )));
        }

        let json: serde_json::Value = response.json().await?;

        // Check for API error
        let ret_code = json["retCode"].as_i64().unwrap_or(-1);
        if ret_code != 0 {
            let msg = json["retMsg"].as_str().unwrap_or("Unknown error");
            return Err(Error::ApiError(format!("Bybit API error: {}", msg)));
        }

        // Parse Bybit response
        // Bybit returns: [[startTime, open, high, low, close, volume, turnover], ...]
        let list = json["result"]["list"]
            .as_array()
            .ok_or_else(|| Error::ParseError("No data in response".to_string()))?;

        let mut bars: Vec<OHLCVBar> = list
            .iter()
            .filter_map(|item| {
                let arr = item.as_array()?;
                let ts = arr.get(0)?.as_str()?.parse::<i64>().ok()?;
                let timestamp = DateTime::from_timestamp_millis(ts)?;

                Some(OHLCVBar {
                    timestamp,
                    open: arr.get(1)?.as_str()?.parse().ok()?,
                    high: arr.get(2)?.as_str()?.parse().ok()?,
                    low: arr.get(3)?.as_str()?.parse().ok()?,
                    close: arr.get(4)?.as_str()?.parse().ok()?,
                    volume: arr.get(5)?.as_str()?.parse().ok()?,
                })
            })
            .collect();

        // Bybit returns newest first, reverse to chronological
        bars.reverse();

        Ok(bars)
    }

    fn source(&self) -> DataSource {
        DataSource::Bybit
    }
}

/// Mock data loader for testing
pub struct MockLoader {
    data: HashMap<String, Vec<OHLCVBar>>,
}

impl MockLoader {
    /// Create new mock loader
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Add mock data for a symbol
    pub fn add_data(&mut self, symbol: &str, bars: Vec<OHLCVBar>) {
        self.data.insert(symbol.to_string(), bars);
    }

    /// Generate random walk data
    pub fn generate_random_walk(
        symbol: &str,
        start: DateTime<Utc>,
        n_bars: usize,
        interval: Interval,
        initial_price: f64,
    ) -> Vec<OHLCVBar> {
        use rand::prelude::*;
        let mut rng = rand::thread_rng();

        let mut bars = Vec::with_capacity(n_bars);
        let mut price = initial_price;

        for i in 0..n_bars {
            let timestamp = start + Duration::seconds(interval.to_seconds() * i as i64);

            // Random walk
            let return_pct = rng.gen_range(-0.02..0.02);
            let open = price;
            price *= 1.0 + return_pct;
            let close = price;

            let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
            let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
            let volume = rng.gen_range(100_000.0..1_000_000.0);

            bars.push(OHLCVBar {
                timestamp,
                open,
                high,
                low,
                close,
                volume,
            });
        }

        bars
    }
}

impl Default for MockLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DataLoader for MockLoader {
    async fn fetch_ohlcv(
        &self,
        symbol: &str,
        interval: Interval,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<OHLCVBar>> {
        // Return cached data if available
        if let Some(bars) = self.data.get(symbol) {
            let filtered: Vec<OHLCVBar> = bars
                .iter()
                .filter(|b| b.timestamp >= start && b.timestamp <= end)
                .cloned()
                .collect();
            return Ok(filtered);
        }

        // Generate random data
        let n_bars = ((end - start).num_seconds() / interval.to_seconds()) as usize;
        let bars = Self::generate_random_walk(
            symbol,
            start,
            n_bars.min(1000),
            interval,
            100.0,
        );

        Ok(bars)
    }

    fn source(&self) -> DataSource {
        DataSource::Mock
    }
}

/// Calculate common technical features from OHLCV data
pub fn calculate_features(bars: &[OHLCVBar]) -> HashMap<String, Vec<f64>> {
    let mut features = HashMap::new();

    if bars.is_empty() {
        return features;
    }

    // Returns
    let returns: Vec<f64> = bars
        .windows(2)
        .map(|w| (w[1].close - w[0].close) / w[0].close)
        .collect();
    features.insert("returns".to_string(), returns.clone());

    // Log returns
    let log_returns: Vec<f64> = bars
        .windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect();
    features.insert("log_returns".to_string(), log_returns);

    // Simple moving averages
    for window in [5, 10, 20, 50] {
        if bars.len() >= window {
            let sma: Vec<f64> = bars
                .windows(window)
                .map(|w| w.iter().map(|b| b.close).sum::<f64>() / window as f64)
                .collect();
            features.insert(format!("sma_{}", window), sma);
        }
    }

    // Volatility (rolling std of returns)
    if returns.len() >= 20 {
        let vol: Vec<f64> = returns
            .windows(20)
            .map(|w| {
                let mean = w.iter().sum::<f64>() / w.len() as f64;
                let variance = w.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / w.len() as f64;
                variance.sqrt()
            })
            .collect();
        features.insert("volatility_20".to_string(), vol);
    }

    // Volume features
    let volumes: Vec<f64> = bars.iter().map(|b| b.volume).collect();
    features.insert("volume".to_string(), volumes.clone());

    if volumes.len() >= 20 {
        let vol_sma: Vec<f64> = volumes
            .windows(20)
            .map(|w| w.iter().sum::<f64>() / w.len() as f64)
            .collect();
        features.insert("volume_sma_20".to_string(), vol_sma);
    }

    // Price range
    let ranges: Vec<f64> = bars
        .iter()
        .map(|b| (b.high - b.low) / b.close)
        .collect();
    features.insert("range_pct".to_string(), ranges);

    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_conversions() {
        assert_eq!(Interval::Day1.to_yahoo_interval(), "1d");
        assert_eq!(Interval::Hour1.to_bybit_interval(), "60");
        assert_eq!(Interval::Day1.to_seconds(), 86400);
    }

    #[tokio::test]
    async fn test_mock_loader() {
        let loader = MockLoader::new();
        let start = Utc::now() - Duration::days(30);
        let end = Utc::now();

        let bars = loader
            .fetch_ohlcv("AAPL", Interval::Day1, start, end)
            .await
            .unwrap();

        assert!(!bars.is_empty());
        assert!(bars.len() <= 30);
    }

    #[test]
    fn test_calculate_features() {
        let start = Utc::now() - Duration::days(100);
        let bars = MockLoader::generate_random_walk(
            "TEST",
            start,
            100,
            Interval::Day1,
            100.0,
        );

        let features = calculate_features(&bars);

        assert!(features.contains_key("returns"));
        assert!(features.contains_key("sma_20"));
        assert!(features.contains_key("volatility_20"));
    }
}
