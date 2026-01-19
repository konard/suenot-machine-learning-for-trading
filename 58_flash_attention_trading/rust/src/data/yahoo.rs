//! Yahoo Finance data loading for stock market data.
//!
//! Fetches OHLCV data from Yahoo Finance.

use super::OhlcvData;
use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use serde::Deserialize;

/// Yahoo Finance API response
#[derive(Debug, Deserialize)]
struct YahooResponse {
    chart: ChartResult,
}

#[derive(Debug, Deserialize)]
struct ChartResult {
    result: Option<Vec<ChartData>>,
    error: Option<YahooError>,
}

#[derive(Debug, Deserialize)]
struct YahooError {
    code: String,
    description: String,
}

#[derive(Debug, Deserialize)]
struct ChartData {
    timestamp: Vec<i64>,
    indicators: Indicators,
}

#[derive(Debug, Deserialize)]
struct Indicators {
    quote: Vec<QuoteData>,
}

#[derive(Debug, Deserialize)]
struct QuoteData {
    open: Vec<Option<f64>>,
    high: Vec<Option<f64>>,
    low: Vec<Option<f64>>,
    close: Vec<Option<f64>>,
    volume: Vec<Option<i64>>,
}

/// Yahoo Finance client
pub struct YahooClient {
    base_url: String,
    client: reqwest::blocking::Client,
}

impl Default for YahooClient {
    fn default() -> Self {
        Self::new()
    }
}

impl YahooClient {
    /// Create a new Yahoo Finance client
    pub fn new() -> Self {
        Self {
            base_url: "https://query1.finance.yahoo.com".to_string(),
            client: reqwest::blocking::Client::builder()
                .user_agent("Mozilla/5.0")
                .build()
                .unwrap(),
        }
    }

    /// Fetch historical data
    ///
    /// # Arguments
    /// * `symbol` - Stock ticker (e.g., "AAPL", "MSFT")
    /// * `interval` - Data interval ("1d", "1h", "5m", etc.)
    /// * `range` - Time range ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max")
    ///
    /// # Example
    /// ```rust,ignore
    /// let client = YahooClient::new();
    /// let data = client.fetch("AAPL", "1d", "1y")?;
    /// ```
    pub fn fetch(&self, symbol: &str, interval: &str, range: &str) -> Result<Vec<OhlcvData>> {
        let url = format!(
            "{}/v8/finance/chart/{}?interval={}&range={}",
            self.base_url, symbol, interval, range
        );

        log::info!("Fetching Yahoo data from: {}", url);

        let response: YahooResponse = self
            .client
            .get(&url)
            .send()
            .context("Failed to send request")?
            .json()
            .context("Failed to parse response")?;

        if let Some(error) = response.chart.error {
            anyhow::bail!("Yahoo API error: {} - {}", error.code, error.description);
        }

        let result = response
            .chart
            .result
            .context("No data in response")?
            .into_iter()
            .next()
            .context("Empty result array")?;

        let quote = result
            .indicators
            .quote
            .into_iter()
            .next()
            .context("No quote data")?;

        let mut ohlcv_data = Vec::new();

        for (i, &ts) in result.timestamp.iter().enumerate() {
            let open = quote.open.get(i).and_then(|v| *v);
            let high = quote.high.get(i).and_then(|v| *v);
            let low = quote.low.get(i).and_then(|v| *v);
            let close = quote.close.get(i).and_then(|v| *v);
            let volume = quote.volume.get(i).and_then(|v| *v);

            if let (Some(o), Some(h), Some(l), Some(c), Some(v)) = (open, high, low, close, volume)
            {
                ohlcv_data.push(OhlcvData {
                    timestamp: Utc.timestamp_opt(ts, 0).unwrap(),
                    open: o,
                    high: h,
                    low: l,
                    close: c,
                    volume: v as f64,
                });
            }
        }

        log::info!("Fetched {} data points for {}", ohlcv_data.len(), symbol);

        Ok(ohlcv_data)
    }

    /// Fetch data between specific dates
    pub fn fetch_range(
        &self,
        symbol: &str,
        interval: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<OhlcvData>> {
        let url = format!(
            "{}/v8/finance/chart/{}?interval={}&period1={}&period2={}",
            self.base_url,
            symbol,
            interval,
            start.timestamp(),
            end.timestamp()
        );

        log::info!("Fetching Yahoo data from: {}", url);

        let response: YahooResponse = self
            .client
            .get(&url)
            .send()
            .context("Failed to send request")?
            .json()
            .context("Failed to parse response")?;

        if let Some(error) = response.chart.error {
            anyhow::bail!("Yahoo API error: {} - {}", error.code, error.description);
        }

        let result = response
            .chart
            .result
            .context("No data in response")?
            .into_iter()
            .next()
            .context("Empty result array")?;

        let quote = result
            .indicators
            .quote
            .into_iter()
            .next()
            .context("No quote data")?;

        let mut ohlcv_data = Vec::new();

        for (i, &ts) in result.timestamp.iter().enumerate() {
            let open = quote.open.get(i).and_then(|v| *v);
            let high = quote.high.get(i).and_then(|v| *v);
            let low = quote.low.get(i).and_then(|v| *v);
            let close = quote.close.get(i).and_then(|v| *v);
            let volume = quote.volume.get(i).and_then(|v| *v);

            if let (Some(o), Some(h), Some(l), Some(c), Some(v)) = (open, high, low, close, volume)
            {
                ohlcv_data.push(OhlcvData {
                    timestamp: Utc.timestamp_opt(ts, 0).unwrap(),
                    open: o,
                    high: h,
                    low: l,
                    close: c,
                    volume: v as f64,
                });
            }
        }

        Ok(ohlcv_data)
    }
}

/// Convenience function to fetch Yahoo data
pub fn fetch_yahoo_data(symbol: &str, interval: &str, range: &str) -> Result<Vec<OhlcvData>> {
    let client = YahooClient::new();
    client.fetch(symbol, interval, range)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require network access
    // They are ignored by default to avoid CI failures

    #[test]
    #[ignore]
    fn test_fetch_yahoo_data() {
        let client = YahooClient::new();
        let data = client.fetch("AAPL", "1d", "5d").unwrap();
        assert!(!data.is_empty());
    }
}
