//! Bybit API client implementation

use super::types::*;
use anyhow::{Context, Result};
use std::time::Duration;

const BYBIT_API_URL: &str = "https://api.bybit.com";

/// Bybit API client for fetching market data
#[derive(Debug, Clone)]
pub struct BybitClient {
    client: reqwest::blocking::Client,
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
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: BYBIT_API_URL.to_string(),
        }
    }

    /// Create a client with custom base URL (for testing)
    pub fn with_base_url(base_url: &str) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: base_url.to_string(),
        }
    }

    /// Get available trading instruments
    pub fn get_instruments(&self, category: &str) -> Result<Vec<InstrumentInfo>> {
        let url = format!(
            "{}/v5/market/instruments-info?category={}",
            self.base_url, category
        );

        let response: BybitResponse<InstrumentsResult> = self
            .client
            .get(&url)
            .send()
            .context("Failed to send request")?
            .json()
            .context("Failed to parse response")?;

        if response.ret_code != 0 {
            anyhow::bail!("API error: {}", response.ret_msg);
        }

        Ok(response.result.list)
    }

    /// Get USDT perpetual futures instruments
    pub fn get_usdt_perpetuals(&self) -> Result<Vec<InstrumentInfo>> {
        self.get_instruments("linear")
    }

    /// Get spot trading instruments
    pub fn get_spot_instruments(&self) -> Result<Vec<InstrumentInfo>> {
        self.get_instruments("spot")
    }

    /// Get 24h tickers for all symbols
    pub fn get_tickers(&self, category: &str) -> Result<Vec<TickerInfo>> {
        let url = format!("{}/v5/market/tickers?category={}", self.base_url, category);

        let response: BybitResponse<TickersResult> = self
            .client
            .get(&url)
            .send()
            .context("Failed to send request")?
            .json()
            .context("Failed to parse response")?;

        if response.ret_code != 0 {
            anyhow::bail!("API error: {}", response.ret_msg);
        }

        Ok(response.result.list)
    }

    /// Get top N symbols by 24h turnover
    pub fn get_top_symbols_by_turnover(
        &self,
        category: &str,
        n: usize,
    ) -> Result<Vec<String>> {
        let mut tickers = self.get_tickers(category)?;

        // Sort by 24h turnover (descending)
        tickers.sort_by(|a, b| {
            b.turnover_24h_f64()
                .partial_cmp(&a.turnover_24h_f64())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Filter USDT pairs and take top N
        let symbols: Vec<String> = tickers
            .into_iter()
            .filter(|t| t.symbol.ends_with("USDT"))
            .take(n)
            .map(|t| t.symbol)
            .collect();

        Ok(symbols)
    }

    /// Fetch kline (candlestick) data for a symbol
    pub fn get_klines(
        &self,
        category: &str,
        symbol: &str,
        timeframe: Timeframe,
        limit: Option<u32>,
        start_time: Option<i64>,
        end_time: Option<i64>,
    ) -> Result<Vec<OHLCV>> {
        let mut url = format!(
            "{}/v5/market/kline?category={}&symbol={}&interval={}",
            self.base_url,
            category,
            symbol,
            timeframe.as_str()
        );

        if let Some(limit) = limit {
            url.push_str(&format!("&limit={}", limit.min(1000)));
        }
        if let Some(start) = start_time {
            url.push_str(&format!("&start={}", start));
        }
        if let Some(end) = end_time {
            url.push_str(&format!("&end={}", end));
        }

        let response: BybitResponse<KlineResult> = self
            .client
            .get(&url)
            .send()
            .context("Failed to send request")?
            .json()
            .context("Failed to parse response")?;

        if response.ret_code != 0 {
            anyhow::bail!("API error: {}", response.ret_msg);
        }

        // Convert to OHLCV and reverse (API returns newest first)
        let mut ohlcv: Vec<OHLCV> = response
            .result
            .list
            .into_iter()
            .map(OHLCV::from)
            .collect();
        ohlcv.reverse();

        Ok(ohlcv)
    }

    /// Fetch historical klines with pagination for longer periods
    pub fn get_historical_klines(
        &self,
        category: &str,
        symbol: &str,
        timeframe: Timeframe,
        start_time: i64,
        end_time: i64,
    ) -> Result<Vec<OHLCV>> {
        let mut all_klines = Vec::new();
        let mut current_end = end_time;
        let interval_ms = timeframe.to_minutes() * 60 * 1000;

        while current_end > start_time {
            let klines = self.get_klines(
                category,
                symbol,
                timeframe,
                Some(1000),
                Some(start_time),
                Some(current_end),
            )?;

            if klines.is_empty() {
                break;
            }

            let oldest_timestamp = klines.first().map(|k| k.timestamp).unwrap_or(0);

            for kline in klines {
                if kline.timestamp >= start_time && kline.timestamp <= end_time {
                    all_klines.push(kline);
                }
            }

            // Move end time to before the oldest kline
            current_end = oldest_timestamp - interval_ms;

            // Small delay to avoid rate limiting
            std::thread::sleep(Duration::from_millis(100));
        }

        // Sort by timestamp and remove duplicates
        all_klines.sort_by_key(|k| k.timestamp);
        all_klines.dedup_by_key(|k| k.timestamp);

        Ok(all_klines)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires network access
    fn test_get_tickers() {
        let client = BybitClient::new();
        let tickers = client.get_tickers("linear").unwrap();
        assert!(!tickers.is_empty());
    }

    #[test]
    #[ignore]
    fn test_get_top_symbols() {
        let client = BybitClient::new();
        let symbols = client.get_top_symbols_by_turnover("linear", 10).unwrap();
        assert_eq!(symbols.len(), 10);
        assert!(symbols.iter().all(|s| s.ends_with("USDT")));
    }

    #[test]
    #[ignore]
    fn test_get_klines() {
        let client = BybitClient::new();
        let klines = client
            .get_klines("linear", "BTCUSDT", Timeframe::Day1, Some(100), None, None)
            .unwrap();
        assert!(!klines.is_empty());
        assert!(klines.len() <= 100);
    }
}
