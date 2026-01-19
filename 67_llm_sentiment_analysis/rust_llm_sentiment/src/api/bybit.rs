//! # Bybit API Client
//!
//! Client for fetching cryptocurrency market data from Bybit exchange.
//! Supports historical candlestick data, ticker information, and symbol listing.

use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;
use tokio::time::sleep;

/// Bybit API base URL
const BYBIT_BASE_URL: &str = "https://api.bybit.com/v5";

/// Rate limit delay between requests (milliseconds)
const RATE_LIMIT_DELAY_MS: u64 = 500;

/// Maximum candles per request
const MAX_CANDLES_PER_REQUEST: usize = 1000;

/// Errors that can occur when using the Bybit API
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),

    #[error("API error: {message} (code: {code})")]
    ApiError { code: i32, message: String },

    #[error("Invalid response format: {0}")]
    InvalidResponse(String),

    #[error("Rate limited, please wait")]
    RateLimited,
}

/// Candlestick interval/timeframe
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interval {
    OneMinute,
    ThreeMinutes,
    FiveMinutes,
    FifteenMinutes,
    ThirtyMinutes,
    OneHour,
    TwoHours,
    FourHours,
    SixHours,
    TwelveHours,
    OneDay,
    OneWeek,
    OneMonth,
}

impl Interval {
    /// Convert interval to API string
    pub fn as_str(&self) -> &'static str {
        match self {
            Interval::OneMinute => "1",
            Interval::ThreeMinutes => "3",
            Interval::FiveMinutes => "5",
            Interval::FifteenMinutes => "15",
            Interval::ThirtyMinutes => "30",
            Interval::OneHour => "60",
            Interval::TwoHours => "120",
            Interval::FourHours => "240",
            Interval::SixHours => "360",
            Interval::TwelveHours => "720",
            Interval::OneDay => "D",
            Interval::OneWeek => "W",
            Interval::OneMonth => "M",
        }
    }

    /// Get interval duration in milliseconds
    pub fn duration_ms(&self) -> i64 {
        match self {
            Interval::OneMinute => 60_000,
            Interval::ThreeMinutes => 180_000,
            Interval::FiveMinutes => 300_000,
            Interval::FifteenMinutes => 900_000,
            Interval::ThirtyMinutes => 1_800_000,
            Interval::OneHour => 3_600_000,
            Interval::TwoHours => 7_200_000,
            Interval::FourHours => 14_400_000,
            Interval::SixHours => 21_600_000,
            Interval::TwelveHours => 43_200_000,
            Interval::OneDay => 86_400_000,
            Interval::OneWeek => 604_800_000,
            Interval::OneMonth => 2_592_000_000,
        }
    }
}

/// Candlestick (OHLCV) data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Unix timestamp in milliseconds
    pub timestamp: i64,
    /// Opening price
    pub open: f64,
    /// Highest price
    pub high: f64,
    /// Lowest price
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Trading volume (in base currency)
    pub volume: f64,
    /// Turnover (volume * price, in quote currency)
    pub turnover: f64,
}

/// Market ticker information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Trading symbol
    pub symbol: String,
    /// Last traded price
    pub last_price: f64,
    /// Best bid price
    pub bid_price: f64,
    /// Best ask price
    pub ask_price: f64,
    /// 24-hour trading volume
    pub volume_24h: f64,
    /// 24-hour price change percentage
    pub price_change_percent_24h: f64,
    /// 24-hour high price
    pub high_24h: f64,
    /// 24-hour low price
    pub low_24h: f64,
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

/// Kline response data
#[derive(Debug, Deserialize)]
struct KlineResult {
    list: Vec<Vec<String>>,
}

/// Ticker response data
#[derive(Debug, Deserialize)]
struct TickerResult {
    list: Vec<TickerItem>,
}

#[derive(Debug, Deserialize)]
struct TickerItem {
    symbol: String,
    #[serde(rename = "lastPrice")]
    last_price: String,
    #[serde(rename = "bid1Price")]
    bid1_price: String,
    #[serde(rename = "ask1Price")]
    ask1_price: String,
    #[serde(rename = "volume24h")]
    volume_24h: String,
    #[serde(rename = "price24hPcnt")]
    price_24h_pcnt: String,
    #[serde(rename = "highPrice24h")]
    high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    low_price_24h: String,
}

/// Symbols response data
#[derive(Debug, Deserialize)]
struct SymbolsResult {
    list: Vec<SymbolInfo>,
}

#[derive(Debug, Deserialize)]
struct SymbolInfo {
    symbol: String,
}

/// Bybit API client
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
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: BYBIT_BASE_URL.to_string(),
        }
    }

    /// Fetch candlestick data for a symbol
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Candlestick interval
    /// * `start_time` - Start time (inclusive)
    /// * `end_time` - End time (exclusive)
    ///
    /// # Returns
    /// Vector of candles sorted by timestamp (oldest first)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: Interval,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<Candle>, BybitError> {
        let mut all_candles = Vec::new();
        let mut current_start = start_time.timestamp_millis();
        let end_ms = end_time.timestamp_millis();

        while current_start < end_ms {
            let url = format!(
                "{}/market/kline?category=spot&symbol={}&interval={}&start={}&end={}&limit={}",
                self.base_url,
                symbol,
                interval.as_str(),
                current_start,
                end_ms,
                MAX_CANDLES_PER_REQUEST
            );

            let response: ApiResponse<KlineResult> = self.client
                .get(&url)
                .send()
                .await?
                .json()
                .await?;

            if response.ret_code != 0 {
                return Err(BybitError::ApiError {
                    code: response.ret_code,
                    message: response.ret_msg,
                });
            }

            if response.result.list.is_empty() {
                break;
            }

            for item in response.result.list {
                if item.len() < 7 {
                    continue;
                }

                let parse_i64 = |value: &str, field: &str| -> Result<i64, BybitError> {
                    value.parse().map_err(|_| {
                        BybitError::InvalidResponse(format!("Invalid {}: {}", field, value))
                    })
                };
                let parse_f64 = |value: &str, field: &str| -> Result<f64, BybitError> {
                    value.parse().map_err(|_| {
                        BybitError::InvalidResponse(format!("Invalid {}: {}", field, value))
                    })
                };

                let candle = Candle {
                    timestamp: parse_i64(&item[0], "timestamp")?,
                    open: parse_f64(&item[1], "open")?,
                    high: parse_f64(&item[2], "high")?,
                    low: parse_f64(&item[3], "low")?,
                    close: parse_f64(&item[4], "close")?,
                    volume: parse_f64(&item[5], "volume")?,
                    turnover: parse_f64(&item[6], "turnover")?,
                };

                if candle.timestamp >= current_start && candle.timestamp < end_ms {
                    all_candles.push(candle);
                }
            }

            // Update start time for next request
            if let Some(last) = all_candles.last() {
                current_start = last.timestamp + interval.duration_ms();
            } else {
                break;
            }

            // Rate limiting
            sleep(Duration::from_millis(RATE_LIMIT_DELAY_MS)).await;
        }

        // Sort by timestamp (oldest first)
        all_candles.sort_by_key(|c| c.timestamp);
        all_candles.dedup_by_key(|c| c.timestamp);

        Ok(all_candles)
    }

    /// Get current ticker information for a symbol
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker, BybitError> {
        let url = format!(
            "{}/market/tickers?category=spot&symbol={}",
            self.base_url, symbol
        );

        let response: ApiResponse<TickerResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let item = response.result.list
            .first()
            .ok_or_else(|| BybitError::InvalidResponse("No ticker found".to_string()))?;

        Ok(Ticker {
            symbol: item.symbol.clone(),
            last_price: item.last_price.parse().map_err(|_| {
                BybitError::InvalidResponse(format!("Invalid last_price: {}", item.last_price))
            })?,
            bid_price: item.bid1_price.parse().map_err(|_| {
                BybitError::InvalidResponse(format!("Invalid bid1_price: {}", item.bid1_price))
            })?,
            ask_price: item.ask1_price.parse().map_err(|_| {
                BybitError::InvalidResponse(format!("Invalid ask1_price: {}", item.ask1_price))
            })?,
            volume_24h: item.volume_24h.parse().map_err(|_| {
                BybitError::InvalidResponse(format!("Invalid volume_24h: {}", item.volume_24h))
            })?,
            price_change_percent_24h: item.price_24h_pcnt.parse::<f64>().map_err(|_| {
                BybitError::InvalidResponse(format!("Invalid price_24h_pcnt: {}", item.price_24h_pcnt))
            })? * 100.0,
            high_24h: item.high_price_24h.parse().map_err(|_| {
                BybitError::InvalidResponse(format!("Invalid high_price_24h: {}", item.high_price_24h))
            })?,
            low_24h: item.low_price_24h.parse().map_err(|_| {
                BybitError::InvalidResponse(format!("Invalid low_price_24h: {}", item.low_price_24h))
            })?,
        })
    }

    /// Get list of available trading symbols
    pub async fn get_symbols(&self) -> Result<Vec<String>, BybitError> {
        let url = format!("{}/market/instruments-info?category=spot", self.base_url);

        let response: ApiResponse<SymbolsResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        Ok(response.result.list.into_iter().map(|s| s.symbol).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_as_str() {
        assert_eq!(Interval::OneHour.as_str(), "60");
        assert_eq!(Interval::OneDay.as_str(), "D");
        assert_eq!(Interval::FiveMinutes.as_str(), "5");
    }

    #[test]
    fn test_interval_duration() {
        assert_eq!(Interval::OneMinute.duration_ms(), 60_000);
        assert_eq!(Interval::OneHour.duration_ms(), 3_600_000);
        assert_eq!(Interval::OneDay.duration_ms(), 86_400_000);
    }
}
