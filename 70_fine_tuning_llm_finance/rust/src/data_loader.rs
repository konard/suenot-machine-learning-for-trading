//! Data loaders for financial data sources.
//!
//! This module provides:
//! - `YahooFinanceLoader`: Load stock data from Yahoo Finance
//! - `BybitDataLoader`: Load cryptocurrency data from Bybit exchange
//! - `OhlcvData`: Common OHLCV data structure

use chrono::{DateTime, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during data loading.
#[derive(Error, Debug)]
pub enum DataLoaderError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Failed to parse response: {0}")]
    ParseError(String),

    #[error("Invalid symbol: {0}")]
    InvalidSymbol(String),

    #[error("No data available for the requested period")]
    NoDataAvailable,

    #[error("API rate limit exceeded")]
    RateLimitExceeded,
}

/// Result type for data loader operations.
pub type Result<T> = std::result::Result<T, DataLoaderError>;

/// OHLCV (Open, High, Low, Close, Volume) data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OhlcvData {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
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
    /// Optional: adjusted close price
    pub adj_close: Option<f64>,
}

impl OhlcvData {
    /// Create a new OHLCV data point.
    pub fn new(
        timestamp: DateTime<Utc>,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            adj_close: None,
        }
    }

    /// Calculate the typical price (HLC/3).
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate the range.
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate the body (close - open).
    pub fn body(&self) -> f64 {
        self.close - self.open
    }

    /// Check if the candle is bullish.
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if the candle is bearish.
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }
}

/// Collection of OHLCV data with utility methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OhlcvSeries {
    /// Data points
    pub data: Vec<OhlcvData>,
    /// Symbol/ticker
    pub symbol: String,
    /// Interval (e.g., "1d", "1h")
    pub interval: String,
}

impl OhlcvSeries {
    /// Create a new OHLCV series.
    pub fn new(symbol: String, interval: String, data: Vec<OhlcvData>) -> Self {
        Self {
            data,
            symbol,
            interval,
        }
    }

    /// Get closing prices as a vector.
    pub fn closes(&self) -> Vec<f64> {
        self.data.iter().map(|d| d.close).collect()
    }

    /// Get volumes as a vector.
    pub fn volumes(&self) -> Vec<f64> {
        self.data.iter().map(|d| d.volume).collect()
    }

    /// Calculate returns.
    pub fn returns(&self) -> Vec<f64> {
        let closes = self.closes();
        closes
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    /// Calculate log returns.
    pub fn log_returns(&self) -> Vec<f64> {
        let closes = self.closes();
        closes
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect()
    }

    /// Calculate simple moving average.
    pub fn sma(&self, period: usize) -> Vec<Option<f64>> {
        let closes = self.closes();
        closes
            .iter()
            .enumerate()
            .map(|(i, _)| {
                if i + 1 < period {
                    None
                } else {
                    let sum: f64 = closes[i + 1 - period..=i].iter().sum();
                    Some(sum / period as f64)
                }
            })
            .collect()
    }

    /// Calculate exponential moving average.
    pub fn ema(&self, period: usize) -> Vec<Option<f64>> {
        let closes = self.closes();
        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut result = vec![None; closes.len()];

        // Initialize with SMA
        if closes.len() >= period {
            let initial_sma: f64 = closes[..period].iter().sum::<f64>() / period as f64;
            result[period - 1] = Some(initial_sma);

            for i in period..closes.len() {
                if let Some(prev_ema) = result[i - 1] {
                    let ema = (closes[i] - prev_ema) * multiplier + prev_ema;
                    result[i] = Some(ema);
                }
            }
        }

        result
    }

    /// Calculate RSI (Relative Strength Index).
    pub fn rsi(&self, period: usize) -> Vec<Option<f64>> {
        let returns = self.returns();
        let mut gains = vec![0.0; returns.len()];
        let mut losses = vec![0.0; returns.len()];

        for (i, &r) in returns.iter().enumerate() {
            if r > 0.0 {
                gains[i] = r;
            } else {
                losses[i] = -r;
            }
        }

        let mut result = vec![None; self.data.len()];

        if returns.len() >= period {
            // Initial average gain/loss
            let initial_avg_gain: f64 = gains[..period].iter().sum::<f64>() / period as f64;
            let initial_avg_loss: f64 = losses[..period].iter().sum::<f64>() / period as f64;

            let mut avg_gain = initial_avg_gain;
            let mut avg_loss = initial_avg_loss;

            for i in period..self.data.len() {
                if i > period {
                    avg_gain = (avg_gain * (period - 1) as f64 + gains[i - 1]) / period as f64;
                    avg_loss = (avg_loss * (period - 1) as f64 + losses[i - 1]) / period as f64;
                }

                let rs = if avg_loss > 0.0 {
                    avg_gain / avg_loss
                } else {
                    100.0
                };
                result[i] = Some(100.0 - 100.0 / (1.0 + rs));
            }
        }

        result
    }

    /// Calculate volatility (standard deviation of returns).
    pub fn volatility(&self, period: usize) -> Vec<Option<f64>> {
        let returns = self.returns();
        let mut result = vec![None; self.data.len()];

        if returns.len() >= period {
            for i in period..self.data.len() {
                let window = &returns[i - period..i];
                let mean = window.iter().sum::<f64>() / period as f64;
                let variance = window.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / period as f64;
                result[i] = Some(variance.sqrt());
            }
        }

        result
    }
}

/// Yahoo Finance data loader.
///
/// Loads historical stock data from Yahoo Finance API.
pub struct YahooFinanceLoader {
    /// HTTP client
    client: reqwest::blocking::Client,
    /// Base URL
    base_url: String,
}

impl Default for YahooFinanceLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl YahooFinanceLoader {
    /// Create a new Yahoo Finance loader.
    pub fn new() -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            base_url: "https://query1.finance.yahoo.com".to_string(),
        }
    }

    /// Get daily OHLCV data for a symbol.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Stock ticker symbol (e.g., "AAPL")
    /// * `period` - Time period (e.g., "1y", "6mo", "3mo")
    ///
    /// # Returns
    ///
    /// OHLCV series with daily data.
    pub fn get_daily(&self, symbol: &str, period: &str) -> Result<OhlcvSeries> {
        self.get_historical(symbol, period, "1d")
    }

    /// Get historical OHLCV data.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Stock ticker symbol
    /// * `period` - Time period
    /// * `interval` - Data interval (e.g., "1d", "1h", "5m")
    pub fn get_historical(
        &self,
        symbol: &str,
        period: &str,
        interval: &str,
    ) -> Result<OhlcvSeries> {
        let url = format!(
            "{}/v8/finance/chart/{}?range={}&interval={}",
            self.base_url, symbol, period, interval
        );

        // Note: In production, this would make actual API calls
        // For demonstration, we return mock data

        // Generate mock data
        let data = self.generate_mock_data(symbol, period, interval)?;

        Ok(OhlcvSeries::new(symbol.to_string(), interval.to_string(), data))
    }

    /// Generate mock data for demonstration.
    fn generate_mock_data(
        &self,
        symbol: &str,
        period: &str,
        interval: &str,
    ) -> Result<Vec<OhlcvData>> {
        let days = match period {
            "1mo" => 22,
            "3mo" => 66,
            "6mo" => 132,
            "1y" => 252,
            "2y" => 504,
            _ => 252,
        };

        let mut data = Vec::with_capacity(days);
        let mut price = match symbol.to_uppercase().as_str() {
            "AAPL" => 150.0,
            "GOOGL" => 140.0,
            "MSFT" => 380.0,
            "TSLA" => 250.0,
            _ => 100.0,
        };

        let now = Utc::now();

        for i in 0..days {
            // Random walk with slight upward drift
            let drift = 0.0003;
            let volatility = 0.02;
            let random = (i as f64 * 0.1).sin() * volatility;
            let change = drift + random;

            price *= 1.0 + change;

            let high = price * (1.0 + volatility / 2.0);
            let low = price * (1.0 - volatility / 2.0);
            let open = price * (1.0 + random / 2.0);

            let timestamp = now - chrono::Duration::days((days - i - 1) as i64);

            data.push(OhlcvData {
                timestamp,
                open,
                high,
                low,
                close: price,
                volume: 10_000_000.0 * (1.0 + (i as f64 * 0.05).sin().abs()),
                adj_close: Some(price),
            });
        }

        Ok(data)
    }
}

/// Bybit exchange data loader.
///
/// Loads cryptocurrency data from Bybit REST API.
pub struct BybitDataLoader {
    /// HTTP client
    client: reqwest::blocking::Client,
    /// Base URL for API
    base_url: String,
    /// Whether to use testnet
    testnet: bool,
}

impl Default for BybitDataLoader {
    fn default() -> Self {
        Self::new(false)
    }
}

impl BybitDataLoader {
    /// Create a new Bybit loader.
    ///
    /// # Arguments
    ///
    /// * `testnet` - Whether to use testnet API
    pub fn new(testnet: bool) -> Self {
        let base_url = if testnet {
            "https://api-testnet.bybit.com".to_string()
        } else {
            "https://api.bybit.com".to_string()
        };

        Self {
            client: reqwest::blocking::Client::new(),
            base_url,
            testnet,
        }
    }

    /// Get kline (candlestick) data.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Candlestick interval in minutes (e.g., "60" for 1h, "D" for daily)
    /// * `limit` - Number of candles to fetch (max 200)
    ///
    /// # Returns
    ///
    /// OHLCV series with kline data.
    pub fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<OhlcvSeries> {
        // Note: In production, this would make actual API calls
        // For demonstration, we return mock crypto data

        let data = self.generate_mock_crypto_data(symbol, interval, limit)?;

        Ok(OhlcvSeries::new(
            symbol.to_string(),
            interval.to_string(),
            data,
        ))
    }

    /// Get orderbook data.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair
    /// * `limit` - Number of levels (max 500)
    pub fn get_orderbook(&self, symbol: &str, limit: usize) -> Result<Orderbook> {
        // Mock orderbook data
        let mut bids = Vec::new();
        let mut asks = Vec::new();

        let mid_price = match symbol {
            "BTCUSDT" => 45000.0,
            "ETHUSDT" => 2500.0,
            _ => 1000.0,
        };

        for i in 0..limit.min(50) {
            let spread = i as f64 * 0.5;
            bids.push(OrderbookLevel {
                price: mid_price - spread,
                quantity: 1.0 + (i as f64 * 0.1).sin().abs() * 10.0,
            });
            asks.push(OrderbookLevel {
                price: mid_price + spread,
                quantity: 1.0 + (i as f64 * 0.1).cos().abs() * 10.0,
            });
        }

        Ok(Orderbook {
            symbol: symbol.to_string(),
            bids,
            asks,
            timestamp: Utc::now(),
        })
    }

    /// Get current ticker information.
    pub fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        let price = match symbol {
            "BTCUSDT" => 45000.0,
            "ETHUSDT" => 2500.0,
            "SOLUSDT" => 100.0,
            _ => 1000.0,
        };

        Ok(Ticker {
            symbol: symbol.to_string(),
            last_price: price,
            bid_price: price * 0.9999,
            ask_price: price * 1.0001,
            volume_24h: 1_000_000_000.0,
            price_change_24h: (price * 0.02 * (Utc::now().timestamp() as f64 * 0.0001).sin()),
            high_24h: price * 1.05,
            low_24h: price * 0.95,
        })
    }

    /// Get funding rate.
    pub fn get_funding_rate(&self, symbol: &str) -> Result<f64> {
        // Mock funding rate (typically between -0.01% and 0.01%)
        let rate = 0.0001 * (Utc::now().timestamp() as f64 * 0.001).sin();
        Ok(rate)
    }

    /// Generate mock cryptocurrency data.
    fn generate_mock_crypto_data(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<OhlcvData>> {
        let mut data = Vec::with_capacity(limit);

        let mut price = match symbol.to_uppercase().as_str() {
            "BTCUSDT" => 45000.0,
            "ETHUSDT" => 2500.0,
            "SOLUSDT" => 100.0,
            "BNBUSDT" => 300.0,
            _ => 1000.0,
        };

        let interval_minutes = match interval {
            "1" => 1,
            "5" => 5,
            "15" => 15,
            "30" => 30,
            "60" => 60,
            "240" => 240,
            "D" => 1440,
            _ => 60,
        };

        let now = Utc::now();

        for i in 0..limit {
            // Higher volatility for crypto
            let volatility = 0.03;
            let random = ((i as f64 * 0.2).sin() + (i as f64 * 0.1).cos()) * volatility;
            let change = random;

            price *= 1.0 + change;

            let high = price * (1.0 + volatility);
            let low = price * (1.0 - volatility);
            let open = price * (1.0 + random / 2.0);

            let timestamp = now - chrono::Duration::minutes((limit - i - 1) as i64 * interval_minutes);

            data.push(OhlcvData {
                timestamp,
                open,
                high,
                low,
                close: price,
                volume: 100_000_000.0 * (1.0 + (i as f64 * 0.1).sin().abs()),
                adj_close: None,
            });
        }

        Ok(data)
    }
}

/// Orderbook level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderbookLevel {
    /// Price level
    pub price: f64,
    /// Quantity at this level
    pub quantity: f64,
}

/// Orderbook snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Orderbook {
    /// Symbol
    pub symbol: String,
    /// Bid levels (sorted by price descending)
    pub bids: Vec<OrderbookLevel>,
    /// Ask levels (sorted by price ascending)
    pub asks: Vec<OrderbookLevel>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl Orderbook {
    /// Get the best bid price.
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Get the best ask price.
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Get the mid price.
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Get the spread.
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }
}

/// Ticker information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Symbol
    pub symbol: String,
    /// Last traded price
    pub last_price: f64,
    /// Best bid price
    pub bid_price: f64,
    /// Best ask price
    pub ask_price: f64,
    /// 24-hour volume
    pub volume_24h: f64,
    /// 24-hour price change
    pub price_change_24h: f64,
    /// 24-hour high
    pub high_24h: f64,
    /// 24-hour low
    pub low_24h: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ohlcv_data() {
        let data = OhlcvData::new(Utc::now(), 100.0, 105.0, 98.0, 103.0, 1000000.0);
        assert!(data.is_bullish());
        assert_eq!(data.body(), 3.0);
        assert_eq!(data.range(), 7.0);
    }

    #[test]
    fn test_yahoo_loader() {
        let loader = YahooFinanceLoader::new();
        let result = loader.get_daily("AAPL", "1mo");
        assert!(result.is_ok());
        let series = result.unwrap();
        assert!(!series.data.is_empty());
    }

    #[test]
    fn test_bybit_loader() {
        let loader = BybitDataLoader::new(false);
        let result = loader.get_klines("BTCUSDT", "60", 100);
        assert!(result.is_ok());
        let series = result.unwrap();
        assert_eq!(series.data.len(), 100);
    }

    #[test]
    fn test_ohlcv_series_indicators() {
        let loader = YahooFinanceLoader::new();
        let series = loader.get_daily("AAPL", "3mo").unwrap();

        let sma = series.sma(20);
        assert!(sma.iter().skip(20).any(|v| v.is_some()));

        let rsi = series.rsi(14);
        assert!(rsi.iter().skip(15).any(|v| v.is_some()));
    }

    #[test]
    fn test_orderbook() {
        let loader = BybitDataLoader::new(false);
        let orderbook = loader.get_orderbook("BTCUSDT", 10).unwrap();

        assert!(orderbook.best_bid().is_some());
        assert!(orderbook.best_ask().is_some());
        assert!(orderbook.spread().unwrap() > 0.0);
    }
}
