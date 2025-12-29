//! Bybit API client for fetching cryptocurrency market data.
//!
//! # Example
//!
//! ```rust,no_run
//! use bayesian_crypto::data::bybit::{BybitClient, Symbol};
//!
//! #[tokio::main]
//! async fn main() {
//!     let client = BybitClient::new();
//!     let klines = client.get_klines(Symbol::BTCUSDT, "1h", 100).await.unwrap();
//!     println!("Fetched {} candles", klines.len());
//! }
//! ```

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Supported trading symbols on Bybit
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Symbol {
    BTCUSDT,
    ETHUSDT,
    SOLUSDT,
    XRPUSDT,
    DOGEUSDT,
    ADAUSDT,
    AVAXUSDT,
    DOTUSDT,
    LINKUSDT,
    MATICUSDT,
}

impl Symbol {
    /// Get the string representation for API calls
    pub fn as_str(&self) -> &'static str {
        match self {
            Symbol::BTCUSDT => "BTCUSDT",
            Symbol::ETHUSDT => "ETHUSDT",
            Symbol::SOLUSDT => "SOLUSDT",
            Symbol::XRPUSDT => "XRPUSDT",
            Symbol::DOGEUSDT => "DOGEUSDT",
            Symbol::ADAUSDT => "ADAUSDT",
            Symbol::AVAXUSDT => "AVAXUSDT",
            Symbol::DOTUSDT => "DOTUSDT",
            Symbol::LINKUSDT => "LINKUSDT",
            Symbol::MATICUSDT => "MATICUSDT",
        }
    }

    /// Parse symbol from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "BTCUSDT" => Some(Symbol::BTCUSDT),
            "ETHUSDT" => Some(Symbol::ETHUSDT),
            "SOLUSDT" => Some(Symbol::SOLUSDT),
            "XRPUSDT" => Some(Symbol::XRPUSDT),
            "DOGEUSDT" => Some(Symbol::DOGEUSDT),
            "ADAUSDT" => Some(Symbol::ADAUSDT),
            "AVAXUSDT" => Some(Symbol::AVAXUSDT),
            "DOTUSDT" => Some(Symbol::DOTUSDT),
            "LINKUSDT" => Some(Symbol::LINKUSDT),
            "MATICUSDT" => Some(Symbol::MATICUSDT),
            _ => None,
        }
    }

    /// Get all available symbols
    pub fn all() -> Vec<Self> {
        vec![
            Symbol::BTCUSDT,
            Symbol::ETHUSDT,
            Symbol::SOLUSDT,
            Symbol::XRPUSDT,
            Symbol::DOGEUSDT,
            Symbol::ADAUSDT,
            Symbol::AVAXUSDT,
            Symbol::DOTUSDT,
            Symbol::LINKUSDT,
            Symbol::MATICUSDT,
        ]
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Opening timestamp (milliseconds)
    pub timestamp: i64,
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
    /// Get the datetime for this candle
    pub fn datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(self.timestamp)
            .unwrap_or_else(|| Utc::now())
    }

    /// Calculate the return from open to close
    pub fn return_pct(&self) -> f64 {
        (self.close - self.open) / self.open
    }

    /// Check if the candle is bullish (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Calculate the candle body size
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Calculate the full range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }
}

/// Bybit API response structure
#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    symbol: String,
    category: String,
    list: Vec<Vec<String>>,
}

/// Client for interacting with Bybit API
#[derive(Debug, Clone)]
pub struct BybitClient {
    base_url: String,
    client: reqwest::Client,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Create a new Bybit API client
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Create a client with custom base URL (for testnet)
    pub fn with_testnet() -> Self {
        Self {
            base_url: "https://api-testnet.bybit.com".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Fetch kline (candlestick) data from Bybit
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol
    /// * `interval` - Time interval: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, M, W
    /// * `limit` - Number of candles to fetch (max 1000)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use bayesian_crypto::data::bybit::{BybitClient, Symbol};
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let client = BybitClient::new();
    ///     let klines = client.get_klines(Symbol::BTCUSDT, "60", 200).await.unwrap();
    /// }
    /// ```
    pub async fn get_klines(
        &self,
        symbol: Symbol,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url,
            symbol.as_str(),
            interval,
            limit.min(1000)
        );

        let response: BybitResponse = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to send request to Bybit")?
            .json()
            .await
            .context("Failed to parse Bybit response")?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {} - {}", response.ret_code, response.ret_msg);
        }

        let klines: Vec<Kline> = response
            .result
            .list
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 7 {
                    Some(Kline {
                        timestamp: row[0].parse().ok()?,
                        open: row[1].parse().ok()?,
                        high: row[2].parse().ok()?,
                        low: row[3].parse().ok()?,
                        close: row[4].parse().ok()?,
                        volume: row[5].parse().ok()?,
                        turnover: row[6].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first, reverse to get chronological order
        let mut klines = klines;
        klines.reverse();

        Ok(klines)
    }

    /// Fetch klines for multiple symbols in parallel
    pub async fn get_multiple_klines(
        &self,
        symbols: &[Symbol],
        interval: &str,
        limit: u32,
    ) -> Result<Vec<(Symbol, Vec<Kline>)>> {
        let futures: Vec<_> = symbols
            .iter()
            .map(|&symbol| {
                let client = self.clone();
                let interval = interval.to_string();
                async move {
                    let klines = client.get_klines(symbol, &interval, limit).await?;
                    Ok::<_, anyhow::Error>((symbol, klines))
                }
            })
            .collect();

        let results = futures::future::join_all(futures).await;

        results.into_iter().collect()
    }

    /// Get the latest price for a symbol
    pub async fn get_latest_price(&self, symbol: Symbol) -> Result<f64> {
        let klines = self.get_klines(symbol, "1", 1).await?;
        klines
            .first()
            .map(|k| k.close)
            .context("No price data available")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_conversion() {
        assert_eq!(Symbol::BTCUSDT.as_str(), "BTCUSDT");
        assert_eq!(Symbol::from_str("ethusdt"), Some(Symbol::ETHUSDT));
        assert_eq!(Symbol::from_str("invalid"), None);
    }

    #[test]
    fn test_kline_calculations() {
        let kline = Kline {
            timestamp: 1700000000000,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        assert!(kline.is_bullish());
        assert_eq!(kline.body_size(), 5.0);
        assert_eq!(kline.range(), 15.0);
        assert!((kline.return_pct() - 0.05).abs() < 1e-10);
    }
}
