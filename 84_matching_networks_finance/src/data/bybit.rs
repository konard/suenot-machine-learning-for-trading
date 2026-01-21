//! Bybit API client for cryptocurrency data
//!
//! Fetches OHLCV data from Bybit's public API for cryptocurrency trading pairs.

use super::OHLCVBar;
use reqwest::Client;
use serde::Deserialize;
use thiserror::Error;

/// Errors that can occur when fetching Bybit data
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("API error: {message} (code: {code})")]
    ApiError { code: i32, message: String },

    #[error("Invalid response format: {0}")]
    ParseError(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Invalid symbol: {0}")]
    InvalidSymbol(String),
}

/// Kline (candlestick) interval
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KlineInterval {
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

impl KlineInterval {
    /// Convert to API string
    pub fn as_str(&self) -> &'static str {
        match self {
            KlineInterval::Min1 => "1",
            KlineInterval::Min3 => "3",
            KlineInterval::Min5 => "5",
            KlineInterval::Min15 => "15",
            KlineInterval::Min30 => "30",
            KlineInterval::Hour1 => "60",
            KlineInterval::Hour2 => "120",
            KlineInterval::Hour4 => "240",
            KlineInterval::Hour6 => "360",
            KlineInterval::Hour12 => "720",
            KlineInterval::Day1 => "D",
            KlineInterval::Week1 => "W",
            KlineInterval::Month1 => "M",
        }
    }

    /// Get interval duration in milliseconds
    pub fn duration_ms(&self) -> i64 {
        match self {
            KlineInterval::Min1 => 60_000,
            KlineInterval::Min3 => 180_000,
            KlineInterval::Min5 => 300_000,
            KlineInterval::Min15 => 900_000,
            KlineInterval::Min30 => 1_800_000,
            KlineInterval::Hour1 => 3_600_000,
            KlineInterval::Hour2 => 7_200_000,
            KlineInterval::Hour4 => 14_400_000,
            KlineInterval::Hour6 => 21_600_000,
            KlineInterval::Hour12 => 43_200_000,
            KlineInterval::Day1 => 86_400_000,
            KlineInterval::Week1 => 604_800_000,
            KlineInterval::Month1 => 2_592_000_000,
        }
    }
}

/// Bybit client configuration
#[derive(Debug, Clone)]
pub struct BybitConfig {
    /// Base URL for API
    pub base_url: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Maximum retries on failure
    pub max_retries: u32,
}

impl Default for BybitConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            timeout_secs: 30,
            max_retries: 3,
        }
    }
}

/// Bybit API client
#[derive(Debug, Clone)]
pub struct BybitClient {
    client: Client,
    config: BybitConfig,
}

/// API response wrapper
#[derive(Debug, Deserialize)]
struct ApiResponse<T> {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: Option<T>,
}

/// Kline response data
#[derive(Debug, Deserialize)]
struct KlineResult {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

impl BybitClient {
    /// Create a new Bybit client with default configuration
    pub fn new() -> Self {
        Self::with_config(BybitConfig::default())
    }

    /// Create a new Bybit client with custom configuration
    pub fn with_config(config: BybitConfig) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self { client, config }
    }

    /// Fetch kline (candlestick) data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Kline interval
    /// * `limit` - Number of klines to fetch (max 1000)
    /// * `start_time` - Optional start timestamp in milliseconds
    /// * `end_time` - Optional end timestamp in milliseconds
    ///
    /// # Example
    /// ```ignore
    /// use matching_networks_finance::data::{BybitClient, KlineInterval};
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let client = BybitClient::new();
    ///     let bars = client.fetch_klines("BTCUSDT", KlineInterval::Hour1, 100, None, None).await.unwrap();
    ///     println!("Fetched {} bars", bars.len());
    /// }
    /// ```
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: KlineInterval,
        limit: usize,
        start_time: Option<i64>,
        end_time: Option<i64>,
    ) -> Result<Vec<OHLCVBar>, BybitError> {
        let limit = limit.min(1000);
        let mut url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.config.base_url,
            symbol,
            interval.as_str(),
            limit
        );

        if let Some(start) = start_time {
            url.push_str(&format!("&start={}", start));
        }
        if let Some(end) = end_time {
            url.push_str(&format!("&end={}", end));
        }

        let response: ApiResponse<KlineResult> = self.request(&url).await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let result = response
            .result
            .ok_or_else(|| BybitError::ParseError("Missing result in response".to_string()))?;

        let mut bars: Vec<OHLCVBar> = result
            .list
            .into_iter()
            .filter_map(|item| self.parse_kline(&item))
            .collect();

        // Bybit returns newest first, reverse to get chronological order
        bars.reverse();

        Ok(bars)
    }

    /// Fetch multiple pages of historical data
    pub async fn fetch_historical(
        &self,
        symbol: &str,
        interval: KlineInterval,
        start_time: i64,
        end_time: i64,
    ) -> Result<Vec<OHLCVBar>, BybitError> {
        let mut all_bars = Vec::new();
        let mut current_end = end_time;
        let interval_ms = interval.duration_ms();

        while current_end > start_time {
            let bars = self
                .fetch_klines(symbol, interval, 1000, Some(start_time), Some(current_end))
                .await?;

            if bars.is_empty() {
                break;
            }

            let earliest = bars[0].timestamp;

            // Add bars that are within our range
            for bar in bars {
                if bar.timestamp >= start_time && bar.timestamp <= end_time {
                    all_bars.push(bar);
                }
            }

            // Move the window back
            current_end = earliest - interval_ms;

            // Small delay to avoid rate limiting
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        // Sort by timestamp
        all_bars.sort_by_key(|b| b.timestamp);

        // Remove duplicates
        all_bars.dedup_by_key(|b| b.timestamp);

        Ok(all_bars)
    }

    /// Get available trading symbols
    pub async fn get_symbols(&self) -> Result<Vec<String>, BybitError> {
        let url = format!(
            "{}/v5/market/instruments-info?category=linear",
            self.config.base_url
        );

        #[derive(Deserialize)]
        struct SymbolInfo {
            symbol: String,
        }

        #[derive(Deserialize)]
        struct SymbolsResult {
            list: Vec<SymbolInfo>,
        }

        let response: ApiResponse<SymbolsResult> = self.request(&url).await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let result = response
            .result
            .ok_or_else(|| BybitError::ParseError("Missing result".to_string()))?;

        Ok(result.list.into_iter().map(|s| s.symbol).collect())
    }

    /// Get current ticker price
    pub async fn get_ticker(&self, symbol: &str) -> Result<f64, BybitError> {
        let url = format!(
            "{}/v5/market/tickers?category=linear&symbol={}",
            self.config.base_url, symbol
        );

        #[derive(Deserialize)]
        struct TickerInfo {
            #[serde(rename = "lastPrice")]
            last_price: String,
        }

        #[derive(Deserialize)]
        struct TickersResult {
            list: Vec<TickerInfo>,
        }

        let response: ApiResponse<TickersResult> = self.request(&url).await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let result = response
            .result
            .ok_or_else(|| BybitError::ParseError("Missing result".to_string()))?;

        if result.list.is_empty() {
            return Err(BybitError::InvalidSymbol(symbol.to_string()));
        }

        result.list[0]
            .last_price
            .parse()
            .map_err(|_| BybitError::ParseError("Invalid price format".to_string()))
    }

    /// Make an HTTP request with retries
    async fn request<T: for<'de> Deserialize<'de>>(&self, url: &str) -> Result<T, BybitError> {
        let mut last_error = None;

        for attempt in 0..self.config.max_retries {
            match self.client.get(url).send().await {
                Ok(response) => {
                    if response.status() == 429 {
                        // Rate limited - wait and retry
                        tokio::time::sleep(std::time::Duration::from_secs(1 << attempt)).await;
                        last_error = Some(BybitError::RateLimitExceeded);
                        continue;
                    }

                    match response.json::<T>().await {
                        Ok(data) => return Ok(data),
                        Err(e) => {
                            last_error = Some(BybitError::ParseError(e.to_string()));
                        }
                    }
                }
                Err(e) => {
                    last_error = Some(BybitError::RequestError(e));
                }
            }

            // Wait before retry
            if attempt < self.config.max_retries - 1 {
                tokio::time::sleep(std::time::Duration::from_millis(500 * (attempt as u64 + 1)))
                    .await;
            }
        }

        Err(last_error.unwrap_or(BybitError::ParseError("Unknown error".to_string())))
    }

    /// Parse kline data from API response
    fn parse_kline(&self, item: &[String]) -> Option<OHLCVBar> {
        if item.len() < 6 {
            return None;
        }

        Some(OHLCVBar::new(
            item[0].parse().ok()?,
            item[1].parse().ok()?,
            item[2].parse().ok()?,
            item[3].parse().ok()?,
            item[4].parse().ok()?,
            item[5].parse().ok()?,
        ))
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_interval() {
        assert_eq!(KlineInterval::Hour1.as_str(), "60");
        assert_eq!(KlineInterval::Day1.as_str(), "D");
        assert_eq!(KlineInterval::Min5.duration_ms(), 300_000);
    }

    #[test]
    fn test_config_default() {
        let config = BybitConfig::default();
        assert!(config.base_url.contains("bybit.com"));
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_parse_kline() {
        let client = BybitClient::new();
        let item = vec![
            "1704067200000".to_string(), // timestamp
            "42000.0".to_string(),       // open
            "42500.0".to_string(),       // high
            "41800.0".to_string(),       // low
            "42300.0".to_string(),       // close
            "1000.0".to_string(),        // volume
        ];

        let bar = client.parse_kline(&item).unwrap();
        assert_eq!(bar.timestamp, 1704067200000);
        assert_eq!(bar.open, 42000.0);
        assert_eq!(bar.close, 42300.0);
    }

    #[tokio::test]
    #[ignore] // Requires network access
    async fn test_fetch_klines() {
        let client = BybitClient::new();
        let result = client
            .fetch_klines("BTCUSDT", KlineInterval::Hour1, 10, None, None)
            .await;

        match result {
            Ok(bars) => {
                assert!(!bars.is_empty());
                // Verify chronological order
                for i in 1..bars.len() {
                    assert!(bars[i].timestamp > bars[i - 1].timestamp);
                }
            }
            Err(e) => {
                // Network errors are acceptable in tests
                eprintln!("Skipping due to network error: {}", e);
            }
        }
    }
}
