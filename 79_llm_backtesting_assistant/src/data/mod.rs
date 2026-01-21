//! Data fetching module
//!
//! Provides functionality to fetch historical market data from various sources
//! including Bybit (cryptocurrency) and stock market APIs.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// OHLCV (Open, High, Low, Close, Volume) candle data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Candle timestamp
    pub timestamp: DateTime<Utc>,
    /// Opening price
    pub open: f64,
    /// Highest price
    pub high: f64,
    /// Lowest price
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Volume
    pub volume: f64,
}

impl Candle {
    /// Create a new candle
    pub fn new(
        timestamp: DateTime<Utc>,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Calculate the candle's range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Check if this is a bullish candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if this is a bearish candle
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Calculate the body size
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }
}

/// Trait for data fetchers
#[async_trait]
pub trait DataFetcher: Send + Sync {
    /// Fetch historical candle data
    async fn fetch_candles(
        &self,
        symbol: &str,
        interval: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<Candle>>;

    /// Get the data source name
    fn source_name(&self) -> &str;
}

/// Bybit API data fetcher for cryptocurrency markets
pub struct BybitDataFetcher {
    client: reqwest::Client,
    base_url: String,
}

impl BybitDataFetcher {
    /// Create a new Bybit data fetcher
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Create with a custom base URL (for testing)
    pub fn with_base_url(base_url: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url,
        }
    }

    /// Convert interval string to Bybit format
    fn convert_interval<'a>(&self, interval: &'a str) -> &'a str {
        match interval {
            "1m" => "1",
            "5m" => "5",
            "15m" => "15",
            "30m" => "30",
            "1h" => "60",
            "4h" => "240",
            "1d" | "D" => "D",
            "1w" | "W" => "W",
            _ => interval,
        }
    }
}

impl Default for BybitDataFetcher {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DataFetcher for BybitDataFetcher {
    async fn fetch_candles(
        &self,
        symbol: &str,
        interval: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<Candle>> {
        let bybit_interval = self.convert_interval(interval);
        let start_ms = start.timestamp_millis();
        let end_ms = end.timestamp_millis();

        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&start={}&end={}&limit=1000",
            self.base_url, symbol, bybit_interval, start_ms, end_ms
        );

        let response = self.client.get(&url).send().await?;
        let data: BybitKlineResponse = response.json().await?;

        if data.ret_code != 0 {
            return Err(Error::ApiError(format!(
                "Bybit API error: {} - {}",
                data.ret_code, data.ret_msg
            )));
        }

        let candles = data
            .result
            .list
            .into_iter()
            .filter_map(|kline| {
                if kline.len() < 6 {
                    return None;
                }
                let timestamp_ms: i64 = kline[0].parse().ok()?;
                let timestamp = DateTime::from_timestamp_millis(timestamp_ms)?;
                Some(Candle {
                    timestamp,
                    open: kline[1].parse().ok()?,
                    high: kline[2].parse().ok()?,
                    low: kline[3].parse().ok()?,
                    close: kline[4].parse().ok()?,
                    volume: kline[5].parse().ok()?,
                })
            })
            .collect();

        Ok(candles)
    }

    fn source_name(&self) -> &str {
        "Bybit"
    }
}

/// Bybit API response structures
#[derive(Debug, Deserialize)]
struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitKlineResult,
}

#[derive(Debug, Deserialize)]
struct BybitKlineResult {
    list: Vec<Vec<String>>,
}

/// Stock market data fetcher using Yahoo Finance API
pub struct StockDataFetcher {
    client: reqwest::Client,
}

impl StockDataFetcher {
    /// Create a new stock data fetcher
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

impl Default for StockDataFetcher {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DataFetcher for StockDataFetcher {
    async fn fetch_candles(
        &self,
        symbol: &str,
        interval: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<Candle>> {
        // Yahoo Finance API endpoint
        let yf_interval = match interval {
            "1d" | "D" => "1d",
            "1w" | "W" => "1wk",
            "1m" => "1m",
            "5m" => "5m",
            "15m" => "15m",
            "30m" => "30m",
            "1h" | "60" => "1h",
            _ => "1d",
        };

        let url = format!(
            "https://query1.finance.yahoo.com/v8/finance/chart/{}?period1={}&period2={}&interval={}",
            symbol,
            start.timestamp(),
            end.timestamp(),
            yf_interval
        );

        let response = self.client
            .get(&url)
            .header("User-Agent", "Mozilla/5.0")
            .send()
            .await?;

        let data: YahooFinanceResponse = response.json().await?;

        let chart = data
            .chart
            .result
            .into_iter()
            .next()
            .ok_or_else(|| Error::ApiError("No data returned from Yahoo Finance".to_string()))?;

        let timestamps = chart.timestamp;
        let quotes = chart.indicators.quote.into_iter().next().ok_or_else(|| {
            Error::ApiError("No quote data returned from Yahoo Finance".to_string())
        })?;

        let candles = timestamps
            .into_iter()
            .enumerate()
            .filter_map(|(i, ts)| {
                let timestamp = DateTime::from_timestamp(ts, 0)?;
                Some(Candle {
                    timestamp,
                    open: quotes.open.get(i).copied().flatten()?,
                    high: quotes.high.get(i).copied().flatten()?,
                    low: quotes.low.get(i).copied().flatten()?,
                    close: quotes.close.get(i).copied().flatten()?,
                    volume: quotes.volume.get(i).copied().flatten()? as f64,
                })
            })
            .collect();

        Ok(candles)
    }

    fn source_name(&self) -> &str {
        "Yahoo Finance"
    }
}

/// Yahoo Finance API response structures
#[derive(Debug, Deserialize)]
struct YahooFinanceResponse {
    chart: YahooChart,
}

#[derive(Debug, Deserialize)]
struct YahooChart {
    result: Vec<YahooChartResult>,
}

#[derive(Debug, Deserialize)]
struct YahooChartResult {
    timestamp: Vec<i64>,
    indicators: YahooIndicators,
}

#[derive(Debug, Deserialize)]
struct YahooIndicators {
    quote: Vec<YahooQuote>,
}

#[derive(Debug, Deserialize)]
struct YahooQuote {
    open: Vec<Option<f64>>,
    high: Vec<Option<f64>>,
    low: Vec<Option<f64>>,
    close: Vec<Option<f64>>,
    volume: Vec<Option<u64>>,
}

/// Generate sample candle data for testing
pub fn generate_sample_candles(num_candles: usize, start_price: f64) -> Vec<Candle> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut candles = Vec::with_capacity(num_candles);
    let mut current_price = start_price;
    let now = Utc::now();

    for i in 0..num_candles {
        let change_pct = rng.gen_range(-0.03..0.03);
        let open = current_price;
        let close = open * (1.0 + change_pct);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(1000.0..10000.0);

        let timestamp = now - chrono::Duration::days((num_candles - i) as i64);

        candles.push(Candle {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        });

        current_price = close;
    }

    candles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_properties() {
        let candle = Candle::new(
            Utc::now(),
            100.0,
            105.0,
            98.0,
            103.0,
            1000.0,
        );
        assert!(candle.is_bullish());
        assert!(!candle.is_bearish());
        assert!((candle.range() - 7.0).abs() < 0.001);
        assert!((candle.body() - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_sample_candle_generation() {
        let candles = generate_sample_candles(100, 100.0);
        assert_eq!(candles.len(), 100);
        for candle in &candles {
            assert!(candle.high >= candle.low);
            assert!(candle.high >= candle.open);
            assert!(candle.high >= candle.close);
            assert!(candle.low <= candle.open);
            assert!(candle.low <= candle.close);
        }
    }
}
