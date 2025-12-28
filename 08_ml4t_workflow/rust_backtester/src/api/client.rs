//! Bybit API client implementation.

use chrono::{DateTime, TimeZone, Utc};
use reqwest::Client;
use tracing::{debug, info};

use super::error::ApiError;
use super::response::*;
use crate::models::{Candle, Timeframe};

/// Bybit API base URLs.
pub const BYBIT_MAINNET: &str = "https://api.bybit.com";
pub const BYBIT_TESTNET: &str = "https://api-testnet.bybit.com";

/// Bybit API client for fetching market data.
#[derive(Clone)]
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client for mainnet.
    pub fn new() -> Self {
        Self::with_base_url(BYBIT_MAINNET)
    }

    /// Create a new Bybit client for testnet.
    pub fn testnet() -> Self {
        Self::with_base_url(BYBIT_TESTNET)
    }

    /// Create a new Bybit client with custom base URL.
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Fetch kline (candlestick) data.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `timeframe` - Candlestick timeframe
    /// * `limit` - Number of candles to fetch (max 1000)
    /// * `start` - Optional start time
    /// * `end` - Optional end time
    pub async fn get_klines(
        &self,
        symbol: &str,
        timeframe: Timeframe,
        limit: Option<u32>,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Result<Vec<Candle>, ApiError> {
        let mut url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}",
            self.base_url,
            symbol,
            timeframe.to_bybit_interval()
        );

        if let Some(l) = limit {
            url.push_str(&format!("&limit={}", l.min(1000)));
        }

        if let Some(s) = start {
            url.push_str(&format!("&start={}", s.timestamp_millis()));
        }

        if let Some(e) = end {
            url.push_str(&format!("&end={}", e.timestamp_millis()));
        }

        debug!("Fetching klines: {}", url);

        let response: BybitResponse<KlineResult> = self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(ApiError::BybitError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let candles: Vec<Candle> = response
            .result
            .list
            .into_iter()
            .filter_map(|k| {
                let timestamp_ms: i64 = k.0.parse().ok()?;
                let timestamp = Utc.timestamp_millis_opt(timestamp_ms).single()?;
                let open: f64 = k.1.parse().ok()?;
                let high: f64 = k.2.parse().ok()?;
                let low: f64 = k.3.parse().ok()?;
                let close: f64 = k.4.parse().ok()?;
                let volume: f64 = k.5.parse().ok()?;
                let turnover: f64 = k.6.parse().ok()?;

                Some(Candle {
                    timestamp,
                    symbol: symbol.to_string(),
                    open,
                    high,
                    low,
                    close,
                    volume,
                    turnover,
                })
            })
            .collect();

        // Bybit returns newest first, we want oldest first
        let mut sorted_candles = candles;
        sorted_candles.sort_by_key(|c| c.timestamp);

        info!(
            "Fetched {} candles for {} {}",
            sorted_candles.len(),
            symbol,
            timeframe
        );

        Ok(sorted_candles)
    }

    /// Fetch historical klines with pagination.
    ///
    /// This method handles the 1000 candle limit by making multiple requests.
    pub async fn get_historical_klines(
        &self,
        symbol: &str,
        timeframe: Timeframe,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<Candle>, ApiError> {
        let mut all_candles = Vec::new();
        let mut current_end = end;
        let interval_ms = timeframe.as_seconds() * 1000;

        loop {
            let candles = self
                .get_klines(symbol, timeframe, Some(1000), Some(start), Some(current_end))
                .await?;

            if candles.is_empty() {
                break;
            }

            let oldest_timestamp = candles.first().unwrap().timestamp;
            all_candles.extend(candles);

            if oldest_timestamp <= start {
                break;
            }

            // Move end time to before the oldest candle
            current_end = oldest_timestamp - chrono::Duration::milliseconds(interval_ms);

            // Small delay to avoid rate limiting
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        // Sort by timestamp and remove duplicates
        all_candles.sort_by_key(|c| c.timestamp);
        all_candles.dedup_by_key(|c| c.timestamp);

        // Filter to requested range
        all_candles.retain(|c| c.timestamp >= start && c.timestamp <= end);

        info!(
            "Fetched {} total candles for {} {} from {} to {}",
            all_candles.len(),
            symbol,
            timeframe,
            start,
            end
        );

        Ok(all_candles)
    }

    /// Get current ticker information.
    pub async fn get_ticker(&self, symbol: &str) -> Result<TickerInfo, ApiError> {
        let url = format!(
            "{}/v5/market/tickers?category=spot&symbol={}",
            self.base_url, symbol
        );

        debug!("Fetching ticker: {}", url);

        let response: BybitResponse<TickersResult> =
            self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(ApiError::BybitError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        response
            .result
            .list
            .into_iter()
            .next()
            .ok_or(ApiError::NoData)
    }

    /// Get all tickers.
    pub async fn get_all_tickers(&self) -> Result<Vec<TickerInfo>, ApiError> {
        let url = format!("{}/v5/market/tickers?category=spot", self.base_url);

        debug!("Fetching all tickers");

        let response: BybitResponse<TickersResult> =
            self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(ApiError::BybitError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        Ok(response.result.list)
    }

    /// Get instrument information.
    pub async fn get_instruments(&self) -> Result<Vec<InstrumentInfo>, ApiError> {
        let url = format!("{}/v5/market/instruments-info?category=spot", self.base_url);

        debug!("Fetching instruments");

        let response: BybitResponse<InstrumentsResult> =
            self.client.get(&url).send().await?.json().await?;

        if response.ret_code != 0 {
            return Err(ApiError::BybitError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        Ok(response.result.list)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}
