//! Bybit API client for fetching cryptocurrency market data.
//!
//! Provides methods to fetch:
//! - Historical OHLCV (Kline) data
//! - Real-time ticker information
//! - Order book data

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Base URL for Bybit API v5.
const BYBIT_API_BASE: &str = "https://api.bybit.com/v5";

/// OHLCV candlestick data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
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
    /// Trading volume
    pub volume: f64,
    /// Turnover (quote volume)
    pub turnover: f64,
}

impl Kline {
    /// Calculate the candle body size (absolute).
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Calculate the full candle range.
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Check if this is a bullish candle.
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Calculate returns from this candle.
    pub fn returns(&self) -> f64 {
        if self.open == 0.0 {
            return 0.0;
        }
        (self.close - self.open) / self.open
    }
}

/// Real-time ticker information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickerInfo {
    /// Trading symbol
    pub symbol: String,
    /// Last traded price
    pub last_price: f64,
    /// 24h high price
    pub high_price_24h: f64,
    /// 24h low price
    pub low_price_24h: f64,
    /// 24h trading volume
    pub volume_24h: f64,
    /// 24h turnover
    pub turnover_24h: f64,
    /// 24h price change percentage
    pub price_change_24h: f64,
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
    list: Vec<Vec<String>>,
}

/// Ticker response result.
#[derive(Debug, Deserialize)]
struct TickerResult {
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
    price_24h_pcnt: String,
}

/// Bybit API client.
pub struct BybitClient {
    client: reqwest::blocking::Client,
    base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client.
    pub fn new() -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            base_url: BYBIT_API_BASE.to_string(),
        }
    }

    /// Create a client with a custom base URL (for testing).
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Fetch historical kline (OHLCV) data.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Candlestick interval (e.g., "1", "5", "60", "D")
    /// * `limit` - Number of candles to fetch (max 200)
    ///
    /// # Returns
    /// Vector of Kline data sorted by timestamp (oldest first).
    pub fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: Option<u32>,
    ) -> Result<Vec<Kline>> {
        let url = format!("{}/market/kline", self.base_url);
        let limit = limit.unwrap_or(100).min(200);

        let response: BybitResponse<KlineResult> = self
            .client
            .get(&url)
            .query(&[
                ("category", "spot"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.to_string()),
            ])
            .send()
            .context("Failed to send request to Bybit")?
            .json()
            .context("Failed to parse Bybit response")?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let mut klines: Vec<Kline> = response
            .result
            .list
            .into_iter()
            .filter_map(|row| {
                if row.len() < 7 {
                    return None;
                }

                let timestamp_ms: i64 = row[0].parse().ok()?;
                let timestamp =
                    DateTime::from_timestamp_millis(timestamp_ms)?;

                Some(Kline {
                    timestamp,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
                    turnover: row[6].parse().ok()?,
                })
            })
            .collect();

        // Sort by timestamp (oldest first)
        klines.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        Ok(klines)
    }

    /// Fetch current ticker information.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    ///
    /// # Returns
    /// Current ticker information for the symbol.
    pub fn get_ticker(&self, symbol: &str) -> Result<TickerInfo> {
        let url = format!("{}/market/tickers", self.base_url);

        let response: BybitResponse<TickerResult> = self
            .client
            .get(&url)
            .query(&[("category", "spot"), ("symbol", symbol)])
            .send()
            .context("Failed to send request to Bybit")?
            .json()
            .context("Failed to parse Bybit response")?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let item = response
            .result
            .list
            .into_iter()
            .next()
            .context("No ticker data found")?;

        Ok(TickerInfo {
            symbol: item.symbol,
            last_price: item.last_price.parse().unwrap_or(0.0),
            high_price_24h: item.high_price_24h.parse().unwrap_or(0.0),
            low_price_24h: item.low_price_24h.parse().unwrap_or(0.0),
            volume_24h: item.volume_24h.parse().unwrap_or(0.0),
            turnover_24h: item.turnover_24h.parse().unwrap_or(0.0),
            price_change_24h: item.price_24h_pcnt.parse().unwrap_or(0.0),
        })
    }

    /// Fetch multiple klines for different symbols.
    pub fn get_multiple_klines(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: Option<u32>,
    ) -> Result<Vec<(String, Vec<Kline>)>> {
        symbols
            .iter()
            .map(|&symbol| {
                let klines = self.get_klines(symbol, interval, limit)?;
                Ok((symbol.to_string(), klines))
            })
            .collect()
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Time interval enum for Bybit klines.
#[derive(Debug, Clone, Copy)]
pub enum Interval {
    /// 1 minute
    Min1,
    /// 3 minutes
    Min3,
    /// 5 minutes
    Min5,
    /// 15 minutes
    Min15,
    /// 30 minutes
    Min30,
    /// 1 hour
    Hour1,
    /// 2 hours
    Hour2,
    /// 4 hours
    Hour4,
    /// 6 hours
    Hour6,
    /// 12 hours
    Hour12,
    /// 1 day
    Day1,
    /// 1 week
    Week1,
    /// 1 month
    Month1,
}

impl Interval {
    /// Convert to Bybit API interval string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Interval::Min1 => "1",
            Interval::Min3 => "3",
            Interval::Min5 => "5",
            Interval::Min15 => "15",
            Interval::Min30 => "30",
            Interval::Hour1 => "60",
            Interval::Hour2 => "120",
            Interval::Hour4 => "240",
            Interval::Hour6 => "360",
            Interval::Hour12 => "720",
            Interval::Day1 => "D",
            Interval::Week1 => "W",
            Interval::Month1 => "M",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_calculations() {
        let kline = Kline {
            timestamp: Utc::now(),
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        assert_eq!(kline.body_size(), 5.0);
        assert_eq!(kline.range(), 15.0);
        assert!(kline.is_bullish());
        assert!((kline.returns() - 0.05).abs() < 0.0001);
    }

    #[test]
    fn test_interval_as_str() {
        assert_eq!(Interval::Hour1.as_str(), "60");
        assert_eq!(Interval::Day1.as_str(), "D");
        assert_eq!(Interval::Min5.as_str(), "5");
    }
}
