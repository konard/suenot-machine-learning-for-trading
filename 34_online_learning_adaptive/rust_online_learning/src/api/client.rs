//! Bybit Exchange API Client
//!
//! Provides methods to fetch market data from Bybit.

use super::error::{ApiError, ApiResult};
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Bybit API base URL
const BASE_URL: &str = "https://api.bybit.com";

/// Candlestick (OHLCV) data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Timestamp in milliseconds
    pub timestamp: u64,
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

impl Candle {
    /// Calculate return from previous candle
    pub fn return_from(&self, previous: &Candle) -> f64 {
        (self.close - previous.close) / previous.close
    }

    /// Calculate intrabar return
    pub fn intrabar_return(&self) -> f64 {
        (self.close - self.open) / self.open
    }

    /// Calculate range as percentage
    pub fn range_pct(&self) -> f64 {
        (self.high - self.low) / self.low
    }
}

/// Available kline intervals
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interval {
    Min1,
    Min3,
    Min5,
    Min15,
    Min30,
    Hour1,
    Hour2,
    Hour4,
    Hour6,
    Hour12,
    Day1,
    Week1,
    Month1,
}

impl Interval {
    /// Convert interval to API string
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

    /// Parse interval from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "1" | "1m" | "1min" => Some(Interval::Min1),
            "3" | "3m" | "3min" => Some(Interval::Min3),
            "5" | "5m" | "5min" => Some(Interval::Min5),
            "15" | "15m" | "15min" => Some(Interval::Min15),
            "30" | "30m" | "30min" => Some(Interval::Min30),
            "60" | "1h" | "1hour" => Some(Interval::Hour1),
            "120" | "2h" | "2hour" => Some(Interval::Hour2),
            "240" | "4h" | "4hour" => Some(Interval::Hour4),
            "360" | "6h" | "6hour" => Some(Interval::Hour6),
            "720" | "12h" | "12hour" => Some(Interval::Hour12),
            "d" | "1d" | "day" => Some(Interval::Day1),
            "w" | "1w" | "week" => Some(Interval::Week1),
            "m" | "1M" | "month" => Some(Interval::Month1),
            _ => None,
        }
    }

    /// Get interval duration in hours
    pub fn hours(&self) -> f64 {
        match self {
            Interval::Min1 => 1.0 / 60.0,
            Interval::Min3 => 3.0 / 60.0,
            Interval::Min5 => 5.0 / 60.0,
            Interval::Min15 => 15.0 / 60.0,
            Interval::Min30 => 30.0 / 60.0,
            Interval::Hour1 => 1.0,
            Interval::Hour2 => 2.0,
            Interval::Hour4 => 4.0,
            Interval::Hour6 => 6.0,
            Interval::Hour12 => 12.0,
            Interval::Day1 => 24.0,
            Interval::Week1 => 168.0,
            Interval::Month1 => 720.0,
        }
    }
}

/// Response wrapper from Bybit API
#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: T,
}

/// Kline result from Bybit API
#[derive(Debug, Deserialize)]
struct KlineResult {
    #[allow(dead_code)]
    symbol: String,
    #[allow(dead_code)]
    category: String,
    list: Vec<Vec<String>>,
}

/// Bybit API client for fetching market data
#[derive(Debug, Clone)]
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
    /// Create a new Bybit client with default settings
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: BASE_URL.to_string(),
        }
    }

    /// Create a new Bybit client with custom base URL (for testnet)
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Create a testnet client
    pub fn testnet() -> Self {
        Self::with_base_url("https://api-testnet.bybit.com")
    }

    /// Fetch kline (candlestick) data
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Kline interval (e.g., "1h", "4h", "1d")
    /// * `limit` - Number of candles to fetch (max 1000)
    ///
    /// # Returns
    ///
    /// Vector of candles sorted by time (oldest first)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> ApiResult<Vec<Candle>> {
        let interval_enum =
            Interval::from_str(interval).ok_or_else(|| ApiError::InvalidInterval(interval.to_string()))?;

        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url,
            symbol.to_uppercase(),
            interval_enum.as_str(),
            limit.min(1000)
        );

        let response: BybitResponse<KlineResult> =
            self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(ApiError::ApiResponseError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let mut candles: Vec<Candle> = response
            .result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() >= 7 {
                    Some(Candle {
                        timestamp: item[0].parse().ok()?,
                        open: item[1].parse().ok()?,
                        high: item[2].parse().ok()?,
                        low: item[3].parse().ok()?,
                        close: item[4].parse().ok()?,
                        volume: item[5].parse().ok()?,
                        turnover: item[6].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by timestamp (oldest first)
        candles.sort_by_key(|c| c.timestamp);

        Ok(candles)
    }

    /// Fetch kline data with time range
    pub async fn get_klines_range(
        &self,
        symbol: &str,
        interval: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> ApiResult<Vec<Candle>> {
        let interval_enum =
            Interval::from_str(interval).ok_or_else(|| ApiError::InvalidInterval(interval.to_string()))?;

        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&start={}&end={}&limit=1000",
            self.base_url,
            symbol.to_uppercase(),
            interval_enum.as_str(),
            start_time.timestamp_millis(),
            end_time.timestamp_millis()
        );

        let response: BybitResponse<KlineResult> =
            self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(ApiError::ApiResponseError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let mut candles: Vec<Candle> = response
            .result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() >= 7 {
                    Some(Candle {
                        timestamp: item[0].parse().ok()?,
                        open: item[1].parse().ok()?,
                        high: item[2].parse().ok()?,
                        low: item[3].parse().ok()?,
                        close: item[4].parse().ok()?,
                        volume: item[5].parse().ok()?,
                        turnover: item[6].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        candles.sort_by_key(|c| c.timestamp);
        Ok(candles)
    }

    /// Get available trading symbols
    pub async fn get_symbols(&self) -> ApiResult<Vec<String>> {
        #[derive(Deserialize)]
        struct SymbolResult {
            list: Vec<SymbolInfo>,
        }

        #[derive(Deserialize)]
        struct SymbolInfo {
            symbol: String,
        }

        let url = format!("{}/v5/market/instruments-info?category=spot", self.base_url);

        let response: BybitResponse<SymbolResult> =
            self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(ApiError::ApiResponseError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        Ok(response.result.list.into_iter().map(|s| s.symbol).collect())
    }

    /// Get ticker price for a symbol
    pub async fn get_ticker(&self, symbol: &str) -> ApiResult<f64> {
        #[derive(Deserialize)]
        struct TickerResult {
            list: Vec<TickerInfo>,
        }

        #[derive(Deserialize)]
        struct TickerInfo {
            #[serde(rename = "lastPrice")]
            last_price: String,
        }

        let url = format!(
            "{}/v5/market/tickers?category=spot&symbol={}",
            self.base_url,
            symbol.to_uppercase()
        );

        let response: BybitResponse<TickerResult> =
            self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(ApiError::ApiResponseError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        response
            .result
            .list
            .first()
            .ok_or(ApiError::NoData)?
            .last_price
            .parse()
            .map_err(|_| ApiError::ParseError("Failed to parse price".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_parsing() {
        assert_eq!(Interval::from_str("1h"), Some(Interval::Hour1));
        assert_eq!(Interval::from_str("4h"), Some(Interval::Hour4));
        assert_eq!(Interval::from_str("1d"), Some(Interval::Day1));
        assert_eq!(Interval::from_str("invalid"), None);
    }

    #[test]
    fn test_interval_as_str() {
        assert_eq!(Interval::Hour1.as_str(), "60");
        assert_eq!(Interval::Day1.as_str(), "D");
    }

    #[test]
    fn test_candle_calculations() {
        let candle1 = Candle {
            timestamp: 1000,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        let candle2 = Candle {
            timestamp: 2000,
            open: 105.0,
            high: 115.0,
            low: 100.0,
            close: 110.0,
            volume: 1200.0,
            turnover: 120000.0,
        };

        assert!((candle2.return_from(&candle1) - 0.0476).abs() < 0.001);
        assert!((candle1.intrabar_return() - 0.05).abs() < 0.001);
    }
}
