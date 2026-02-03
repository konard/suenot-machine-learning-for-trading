//! Bybit API client for fetching market data

use reqwest::Client;
use super::types::{ApiError, Kline, KlineResponse};

/// Bybit API client
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Create client with custom base URL
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Fetch kline (candlestick) data
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Candle interval ("1", "5", "15", "60", "240", "D")
    /// * `limit` - Number of candles (max 1000)
    /// * `start_time` - Optional start timestamp (ms)
    /// * `end_time` - Optional end timestamp (ms)
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
        start_time: Option<i64>,
        end_time: Option<i64>,
    ) -> Result<Vec<Kline>, ApiError> {
        let mut url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        if let Some(start) = start_time {
            url.push_str(&format!("&start={}", start));
        }
        if let Some(end) = end_time {
            url.push_str(&format!("&end={}", end));
        }

        let response: KlineResponse = self
            .client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(ApiError::ApiResponseError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let mut klines: Vec<Kline> = response
            .result
            .list
            .iter()
            .filter_map(|v| Kline::from_strings(v).ok())
            .collect();

        // API returns newest first, reverse to chronological order
        klines.reverse();

        Ok(klines)
    }

    /// Fetch multiple symbols in parallel
    pub async fn fetch_multiple_symbols(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: u32,
    ) -> Result<Vec<(String, Vec<Kline>)>, ApiError> {
        let futures: Vec<_> = symbols
            .iter()
            .map(|symbol| {
                let symbol = symbol.to_string();
                let client = self.clone();
                let interval = interval.to_string();
                async move {
                    let klines = client.fetch_klines(&symbol, &interval, limit, None, None).await?;
                    Ok::<_, ApiError>((symbol, klines))
                }
            })
            .collect();

        let results = futures::future::try_join_all(futures).await?;
        Ok(results)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for BybitClient {
    fn clone(&self) -> Self {
        Self {
            client: Client::new(),
            base_url: self.base_url.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_from_strings() {
        let values = vec![
            "1704067200000".to_string(),
            "42000.0".to_string(),
            "42500.0".to_string(),
            "41800.0".to_string(),
            "42200.0".to_string(),
            "1000.0".to_string(),
            "42200000.0".to_string(),
        ];

        let kline = Kline::from_strings(&values).unwrap();
        assert_eq!(kline.timestamp, 1704067200000);
        assert!((kline.close - 42200.0).abs() < 0.01);
    }
}
