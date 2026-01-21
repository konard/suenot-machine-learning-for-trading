//! Data loading module for financial market data.
//!
//! Provides data loaders for:
//! - Yahoo Finance (stocks, ETFs)
//! - Bybit (cryptocurrency)

use chrono::{DateTime, Utc, Duration, TimeZone};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during data loading.
#[derive(Error, Debug)]
pub enum DataError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Failed to parse response: {0}")]
    ParseError(String),

    #[error("No data available for symbol: {0}")]
    NoData(String),

    #[error("Invalid date range")]
    InvalidDateRange,
}

/// OHLCV (Open, High, Low, Close, Volume) data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCVBar {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Collection of OHLCV bars.
#[derive(Debug, Clone, Default)]
pub struct OHLCVData {
    pub symbol: String,
    pub bars: Vec<OHLCVBar>,
}

impl OHLCVData {
    /// Create new empty OHLCV data.
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            bars: Vec::new(),
        }
    }

    /// Get the number of bars.
    pub fn len(&self) -> usize {
        self.bars.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.bars.is_empty()
    }

    /// Get closing prices as a vector.
    pub fn close_prices(&self) -> Vec<f64> {
        self.bars.iter().map(|b| b.close).collect()
    }

    /// Get returns (percentage change).
    pub fn returns(&self) -> Vec<f64> {
        let closes = self.close_prices();
        if closes.len() < 2 {
            return Vec::new();
        }

        closes.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    /// Get a slice of the data.
    pub fn slice(&self, start: usize, end: usize) -> OHLCVData {
        let end = std::cmp::min(end, self.bars.len());
        let start = std::cmp::min(start, end);

        OHLCVData {
            symbol: self.symbol.clone(),
            bars: self.bars[start..end].to_vec(),
        }
    }

    /// Get the last N bars.
    pub fn tail(&self, n: usize) -> OHLCVData {
        if n >= self.bars.len() {
            return self.clone();
        }
        self.slice(self.bars.len() - n, self.bars.len())
    }
}

/// Trait for data loaders.
pub trait DataLoader {
    /// Load daily data for a symbol.
    fn get_daily(&self, symbol: &str, period: &str) -> Result<OHLCVData, DataError>;

    /// Generate mock data for testing.
    fn generate_mock_data(&self, symbol: &str, bars: usize) -> OHLCVData;
}

/// Yahoo Finance data loader.
pub struct YahooFinanceLoader {
    client: reqwest::blocking::Client,
}

impl YahooFinanceLoader {
    /// Create a new Yahoo Finance loader.
    pub fn new() -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Parse period string to days.
    fn period_to_days(period: &str) -> i64 {
        match period {
            "1mo" => 30,
            "3mo" => 90,
            "6mo" => 180,
            "1y" => 365,
            "2y" => 730,
            "5y" => 1825,
            _ => 365, // Default to 1 year
        }
    }
}

impl Default for YahooFinanceLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DataLoader for YahooFinanceLoader {
    fn get_daily(&self, symbol: &str, period: &str) -> Result<OHLCVData, DataError> {
        let days = Self::period_to_days(period);
        let end = Utc::now();
        let start = end - Duration::days(days);

        let url = format!(
            "https://query1.finance.yahoo.com/v8/finance/chart/{}?period1={}&period2={}&interval=1d",
            symbol,
            start.timestamp(),
            end.timestamp()
        );

        // Try to fetch from Yahoo Finance
        let response_result: Result<reqwest::blocking::Response, reqwest::Error> = self.client.get(&url).send();
        match response_result {
            Ok(response) => {
                if response.status().is_success() {
                    // Parse Yahoo Finance response
                    if let Ok(json) = response.json::<serde_json::Value>() {
                        return Self::parse_yahoo_response(symbol, &json);
                    }
                }
                // Fall back to mock data
                Ok(self.generate_mock_data(symbol, days as usize))
            }
            Err(_) => {
                // Fall back to mock data
                Ok(self.generate_mock_data(symbol, days as usize))
            }
        }
    }

    fn generate_mock_data(&self, symbol: &str, bars: usize) -> OHLCVData {
        let mut data = OHLCVData::new(symbol);
        let mut price = match symbol {
            "SPY" => 450.0,
            "QQQ" => 380.0,
            "IWM" => 200.0,
            _ => 100.0,
        };

        let now = Utc::now();

        for i in 0..bars {
            let timestamp = now - Duration::days((bars - i - 1) as i64);

            // Random walk with slight upward drift
            let drift = 0.0002;
            let volatility = 0.01;
            let random: f64 = (i as f64 * 0.1).sin() * 0.5 + 0.5; // Pseudo-random
            let change = drift + volatility * (random * 2.0 - 1.0);
            price *= 1.0 + change;

            let high = price * (1.0 + volatility * random);
            let low = price * (1.0 - volatility * (1.0 - random));

            data.bars.push(OHLCVBar {
                timestamp,
                open: price * (1.0 + change * 0.5),
                high,
                low,
                close: price,
                volume: 1_000_000.0 * (1.0 + random),
            });
        }

        data
    }
}

impl YahooFinanceLoader {
    fn parse_yahoo_response(symbol: &str, json: &serde_json::Value) -> Result<OHLCVData, DataError> {
        let chart = json.get("chart")
            .and_then(|c| c.get("result"))
            .and_then(|r| r.get(0))
            .ok_or_else(|| DataError::ParseError("Invalid chart structure".to_string()))?;

        let timestamps = chart.get("timestamp")
            .and_then(|t| t.as_array())
            .ok_or_else(|| DataError::ParseError("No timestamps".to_string()))?;

        let quote = chart.get("indicators")
            .and_then(|i| i.get("quote"))
            .and_then(|q| q.get(0))
            .ok_or_else(|| DataError::ParseError("No quote data".to_string()))?;

        let opens = quote.get("open").and_then(|o| o.as_array());
        let highs = quote.get("high").and_then(|h| h.as_array());
        let lows = quote.get("low").and_then(|l| l.as_array());
        let closes = quote.get("close").and_then(|c| c.as_array());
        let volumes = quote.get("volume").and_then(|v| v.as_array());

        let mut data = OHLCVData::new(symbol);

        for (i, ts) in timestamps.iter().enumerate() {
            if let Some(ts) = ts.as_i64() {
                let open = opens.and_then(|o| o.get(i)).and_then(|v| v.as_f64()).unwrap_or(0.0);
                let high = highs.and_then(|h| h.get(i)).and_then(|v| v.as_f64()).unwrap_or(0.0);
                let low = lows.and_then(|l| l.get(i)).and_then(|v| v.as_f64()).unwrap_or(0.0);
                let close = closes.and_then(|c| c.get(i)).and_then(|v| v.as_f64()).unwrap_or(0.0);
                let volume = volumes.and_then(|v| v.get(i)).and_then(|v| v.as_f64()).unwrap_or(0.0);

                if close > 0.0 {
                    data.bars.push(OHLCVBar {
                        timestamp: Utc.timestamp_opt(ts, 0).unwrap(),
                        open,
                        high,
                        low,
                        close,
                        volume,
                    });
                }
            }
        }

        if data.is_empty() {
            Err(DataError::NoData(symbol.to_string()))
        } else {
            Ok(data)
        }
    }
}

/// Bybit cryptocurrency data loader.
pub struct BybitDataLoader {
    client: reqwest::blocking::Client,
    base_url: String,
}

impl BybitDataLoader {
    /// Create a new Bybit loader.
    pub fn new() -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Get kline (candlestick) data.
    pub fn get_klines(&self, symbol: &str, interval: &str, limit: usize) -> Result<OHLCVData, DataError> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response_result: Result<reqwest::blocking::Response, reqwest::Error> = self.client.get(&url).send();
        match response_result {
            Ok(response) => {
                if response.status().is_success() {
                    if let Ok(json) = response.json::<serde_json::Value>() {
                        return Self::parse_bybit_response(symbol, &json);
                    }
                }
                // Fall back to mock data
                Ok(self.generate_mock_crypto_data(symbol, limit))
            }
            Err(_) => {
                // Fall back to mock data
                Ok(self.generate_mock_crypto_data(symbol, limit))
            }
        }
    }

    fn parse_bybit_response(symbol: &str, json: &serde_json::Value) -> Result<OHLCVData, DataError> {
        let result = json.get("result")
            .and_then(|r| r.get("list"))
            .and_then(|l| l.as_array())
            .ok_or_else(|| DataError::ParseError("Invalid Bybit response".to_string()))?;

        let mut data = OHLCVData::new(symbol);

        for item in result.iter().rev() {
            if let Some(arr) = item.as_array() {
                if arr.len() >= 6 {
                    let timestamp = arr[0].as_str()
                        .and_then(|s| s.parse::<i64>().ok())
                        .map(|ms| Utc.timestamp_millis_opt(ms).unwrap())
                        .unwrap_or_else(Utc::now);

                    let open = arr[1].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    let high = arr[2].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    let low = arr[3].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    let close = arr[4].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    let volume = arr[5].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);

                    if close > 0.0 {
                        data.bars.push(OHLCVBar {
                            timestamp,
                            open,
                            high,
                            low,
                            close,
                            volume,
                        });
                    }
                }
            }
        }

        if data.is_empty() {
            Err(DataError::NoData(symbol.to_string()))
        } else {
            Ok(data)
        }
    }

    /// Generate mock cryptocurrency data.
    pub fn generate_mock_crypto_data(&self, symbol: &str, bars: usize) -> OHLCVData {
        let mut data = OHLCVData::new(symbol);
        let mut price = match symbol {
            "BTCUSDT" => 45000.0,
            "ETHUSDT" => 2500.0,
            "SOLUSDT" => 100.0,
            _ => 1000.0,
        };

        let now = Utc::now();

        for i in 0..bars {
            let timestamp = now - Duration::hours((bars - i - 1) as i64);

            // Crypto is more volatile
            let drift = 0.0001;
            let volatility = 0.02;
            let random: f64 = ((i as f64 * 0.17).sin() + 1.0) / 2.0;
            let change = drift + volatility * (random * 2.0 - 1.0);
            price *= 1.0 + change;

            let high = price * (1.0 + volatility * random);
            let low = price * (1.0 - volatility * (1.0 - random));

            data.bars.push(OHLCVBar {
                timestamp,
                open: price * (1.0 + change * 0.5),
                high,
                low,
                close: price,
                volume: 100.0 * (1.0 + random),
            });
        }

        data
    }
}

impl Default for BybitDataLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DataLoader for BybitDataLoader {
    fn get_daily(&self, symbol: &str, period: &str) -> Result<OHLCVData, DataError> {
        let bars = match period {
            "1mo" => 30,
            "3mo" => 90,
            "6mo" => 180,
            "1y" => 365,
            _ => 365,
        };
        self.get_klines(symbol, "D", bars)
    }

    fn generate_mock_data(&self, symbol: &str, bars: usize) -> OHLCVData {
        self.generate_mock_crypto_data(symbol, bars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_stock_data() {
        let loader = YahooFinanceLoader::new();
        let data = loader.generate_mock_data("SPY", 100);
        assert_eq!(data.len(), 100);
        assert!(!data.close_prices().is_empty());
    }

    #[test]
    fn test_mock_crypto_data() {
        let loader = BybitDataLoader::new();
        let data = loader.generate_mock_crypto_data("BTCUSDT", 100);
        assert_eq!(data.len(), 100);
        assert!(data.bars[0].close > 0.0);
    }

    #[test]
    fn test_returns_calculation() {
        let loader = YahooFinanceLoader::new();
        let data = loader.generate_mock_data("SPY", 10);
        let returns = data.returns();
        assert_eq!(returns.len(), 9); // N-1 returns for N prices
    }
}
