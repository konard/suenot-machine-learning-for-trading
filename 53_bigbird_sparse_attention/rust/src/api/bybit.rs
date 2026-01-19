//! Bybit API client
//!
//! Client for fetching cryptocurrency data from Bybit exchange.

use reqwest::Client;
use std::time::Duration;

use super::types::{BybitKlineResult, BybitResponse, KlineData, KlineInterval, MarketData};

/// Bybit API error
#[derive(Debug, thiserror::Error)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),
    #[error("API error: {code} - {message}")]
    ApiError { code: i32, message: String },
    #[error("Parse error: {0}")]
    ParseError(String),
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
    /// Create a new Bybit client with default settings
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Create a client with custom base URL (for testnet)
    pub fn with_base_url(base_url: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: base_url.to_string(),
        }
    }

    /// Fetch kline (candlestick) data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Kline interval
    /// * `limit` - Number of klines to fetch (max 1000)
    /// * `start` - Start timestamp in milliseconds (optional)
    /// * `end` - End timestamp in milliseconds (optional)
    ///
    /// # Returns
    /// * `MarketData` containing the kline data
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: KlineInterval,
        limit: u32,
        start: Option<i64>,
        end: Option<i64>,
    ) -> Result<MarketData, BybitError> {
        let mut url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url,
            symbol,
            interval.as_str(),
            limit.min(1000)
        );

        if let Some(start_ts) = start {
            url.push_str(&format!("&start={}", start_ts));
        }
        if let Some(end_ts) = end {
            url.push_str(&format!("&end={}", end_ts));
        }

        let response = self.client.get(&url).send().await?;

        let api_response: BybitResponse<BybitKlineResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        let klines = api_response
            .result
            .list
            .into_iter()
            .filter_map(|item| parse_kline(&item).ok())
            .collect::<Vec<_>>();

        // Bybit returns newest first, so reverse for chronological order
        let mut klines = klines;
        klines.reverse();

        Ok(MarketData::new(
            symbol.to_string(),
            interval.to_string(),
            klines,
        ))
    }

    /// Fetch klines for multiple symbols
    pub async fn get_klines_multi(
        &self,
        symbols: &[&str],
        interval: KlineInterval,
        limit: u32,
    ) -> Result<Vec<MarketData>, BybitError> {
        let mut results = Vec::with_capacity(symbols.len());

        for symbol in symbols {
            match self.get_klines(symbol, interval, limit, None, None).await {
                Ok(data) => results.push(data),
                Err(e) => {
                    eprintln!("Warning: Failed to fetch {}: {}", symbol, e);
                }
            }
        }

        Ok(results)
    }

    /// Fetch historical klines with pagination
    pub async fn get_historical_klines(
        &self,
        symbol: &str,
        interval: KlineInterval,
        total_limit: u32,
    ) -> Result<MarketData, BybitError> {
        let mut all_klines = Vec::new();
        let mut end_time: Option<i64> = None;
        let mut remaining = total_limit;

        while remaining > 0 {
            let batch_size = remaining.min(1000);
            let data = self
                .get_klines(symbol, interval, batch_size, None, end_time)
                .await?;

            if data.is_empty() {
                break;
            }

            end_time = Some(data.klines.first().unwrap().timestamp - 1);
            let mut batch = data.klines;
            batch.reverse(); // Reverse to prepend older data
            all_klines.splice(0..0, batch);

            remaining = remaining.saturating_sub(batch_size);

            // Rate limit protection
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Ok(MarketData::new(
            symbol.to_string(),
            interval.to_string(),
            all_klines,
        ))
    }
}

/// Parse a kline from Bybit's string array format
fn parse_kline(item: &[String]) -> Result<KlineData, BybitError> {
    if item.len() < 7 {
        return Err(BybitError::ParseError(
            "Insufficient kline data fields".to_string(),
        ));
    }

    Ok(KlineData {
        timestamp: item[0]
            .parse()
            .map_err(|e| BybitError::ParseError(format!("Invalid timestamp: {}", e)))?,
        open: item[1]
            .parse()
            .map_err(|e| BybitError::ParseError(format!("Invalid open: {}", e)))?,
        high: item[2]
            .parse()
            .map_err(|e| BybitError::ParseError(format!("Invalid high: {}", e)))?,
        low: item[3]
            .parse()
            .map_err(|e| BybitError::ParseError(format!("Invalid low: {}", e)))?,
        close: item[4]
            .parse()
            .map_err(|e| BybitError::ParseError(format!("Invalid close: {}", e)))?,
        volume: item[5]
            .parse()
            .map_err(|e| BybitError::ParseError(format!("Invalid volume: {}", e)))?,
        turnover: item[6]
            .parse()
            .map_err(|e| BybitError::ParseError(format!("Invalid turnover: {}", e)))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_kline() {
        let item = vec![
            "1234567890000".to_string(),
            "50000.0".to_string(),
            "51000.0".to_string(),
            "49000.0".to_string(),
            "50500.0".to_string(),
            "1000.0".to_string(),
            "50000000.0".to_string(),
        ];

        let kline = parse_kline(&item).unwrap();
        assert_eq!(kline.timestamp, 1234567890000);
        assert_eq!(kline.open, 50000.0);
        assert_eq!(kline.high, 51000.0);
        assert_eq!(kline.low, 49000.0);
        assert_eq!(kline.close, 50500.0);
        assert_eq!(kline.volume, 1000.0);
    }

    #[test]
    fn test_client_creation() {
        let client = BybitClient::new();
        assert_eq!(client.base_url, "https://api.bybit.com");

        let testnet_client = BybitClient::with_base_url("https://api-testnet.bybit.com");
        assert_eq!(testnet_client.base_url, "https://api-testnet.bybit.com");
    }
}
