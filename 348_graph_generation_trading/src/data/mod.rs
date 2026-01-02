//! Data module for fetching and preprocessing market data.
//!
//! This module provides:
//! - Bybit API client for fetching OHLCV data
//! - Data preprocessing utilities
//! - Market data structures

mod bybit_client;
mod preprocessor;

pub use bybit_client::{BybitClient, BybitError};
pub use preprocessor::{normalize_returns, calculate_returns, DataPreprocessor};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// OHLCV (Open, High, Low, Close, Volume) candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCV {
    /// Timestamp of the candle
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
    /// Turnover (volume * price)
    pub turnover: f64,
}

impl OHLCV {
    /// Create a new OHLCV candle
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
            turnover: volume * close,
        }
    }

    /// Calculate the typical price (HLC average)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate the candle range
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Check if candle is bullish (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Calculate log return from previous close
    pub fn log_return(&self, prev_close: f64) -> f64 {
        (self.close / prev_close).ln()
    }
}

/// Market data container for multiple symbols
#[derive(Debug, Clone)]
pub struct MarketData {
    /// Symbol names
    pub symbols: Vec<String>,
    /// OHLCV data for each symbol (indexed by symbol order)
    pub data: Vec<Vec<OHLCV>>,
    /// Timeframe (e.g., "1h", "4h", "1d")
    pub timeframe: String,
}

impl MarketData {
    /// Create new MarketData container
    pub fn new(symbols: Vec<String>, timeframe: &str) -> Self {
        let num_symbols = symbols.len();
        Self {
            symbols,
            data: vec![Vec::new(); num_symbols],
            timeframe: timeframe.to_string(),
        }
    }

    /// Get the number of symbols
    pub fn num_symbols(&self) -> usize {
        self.symbols.len()
    }

    /// Get the number of candles (assumes all symbols have same length)
    pub fn num_candles(&self) -> usize {
        self.data.first().map(|d| d.len()).unwrap_or(0)
    }

    /// Get closing prices for all symbols as a 2D array
    /// Returns: Vec<Vec<f64>> where outer is symbols, inner is time
    pub fn close_prices(&self) -> Vec<Vec<f64>> {
        self.data
            .iter()
            .map(|candles| candles.iter().map(|c| c.close).collect())
            .collect()
    }

    /// Get returns for all symbols
    pub fn returns(&self) -> Vec<Vec<f64>> {
        self.data
            .iter()
            .map(|candles| calculate_returns(candles))
            .collect()
    }

    /// Get data for a specific symbol by name
    pub fn get_symbol(&self, symbol: &str) -> Option<&Vec<OHLCV>> {
        self.symbols
            .iter()
            .position(|s| s == symbol)
            .map(|idx| &self.data[idx])
    }

    /// Add candles for a symbol
    pub fn add_candles(&mut self, symbol: &str, candles: Vec<OHLCV>) {
        if let Some(idx) = self.symbols.iter().position(|s| s == symbol) {
            self.data[idx] = candles;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_ohlcv_typical_price() {
        let candle = OHLCV::new(
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            100.0,
            110.0,
            95.0,
            105.0,
            1000.0,
        );

        let expected = (110.0 + 95.0 + 105.0) / 3.0;
        assert!((candle.typical_price() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_ohlcv_is_bullish() {
        let bullish = OHLCV::new(
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            100.0, 110.0, 95.0, 105.0, 1000.0,
        );
        let bearish = OHLCV::new(
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            105.0, 110.0, 95.0, 100.0, 1000.0,
        );

        assert!(bullish.is_bullish());
        assert!(!bearish.is_bullish());
    }

    #[test]
    fn test_market_data() {
        let symbols = vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()];
        let mut data = MarketData::new(symbols, "1h");

        assert_eq!(data.num_symbols(), 2);
        assert_eq!(data.num_candles(), 0);

        let candles = vec![
            OHLCV::new(Utc::now(), 100.0, 110.0, 95.0, 105.0, 1000.0),
            OHLCV::new(Utc::now(), 105.0, 115.0, 100.0, 110.0, 1100.0),
        ];

        data.add_candles("BTCUSDT", candles);
        assert_eq!(data.num_candles(), 2);
    }
}
