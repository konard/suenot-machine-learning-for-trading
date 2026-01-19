//! OHLCV (Open, High, Low, Close, Volume) data structures.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// OHLCV candlestick data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCV {
    /// Timestamp in milliseconds.
    pub timestamp: i64,
    /// Opening price.
    pub open: f64,
    /// Highest price.
    pub high: f64,
    /// Lowest price.
    pub low: f64,
    /// Closing price.
    pub close: f64,
    /// Trading volume.
    pub volume: f64,
}

impl OHLCV {
    /// Create a new OHLCV instance.
    pub fn new(timestamp: i64, open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Calculate the typical price (average of high, low, close).
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate the price range.
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate the return from open to close.
    pub fn return_pct(&self) -> f64 {
        if self.open > 0.0 {
            (self.close - self.open) / self.open * 100.0
        } else {
            0.0
        }
    }

    /// Check if the candle is bullish (close > open).
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if the candle is bearish (close < open).
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }
}

/// Container for OHLCV data from multiple symbols.
#[derive(Debug, Clone, Default)]
pub struct MultiSymbolData {
    /// Map of symbol name to OHLCV data.
    data: HashMap<String, Vec<OHLCV>>,
}

impl MultiSymbolData {
    /// Create a new empty container.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Add data for a symbol.
    pub fn add_symbol(&mut self, symbol: String, ohlcv: Vec<OHLCV>) {
        self.data.insert(symbol, ohlcv);
    }

    /// Get data for a symbol.
    pub fn get(&self, symbol: &str) -> Option<&Vec<OHLCV>> {
        self.data.get(symbol)
    }

    /// Get all symbols.
    pub fn symbols(&self) -> Vec<&String> {
        self.data.keys().collect()
    }

    /// Get the number of symbols.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the container is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Calculate returns for a symbol.
    pub fn calculate_returns(&self, symbol: &str) -> Option<Vec<f64>> {
        self.data.get(symbol).map(|ohlcv| {
            ohlcv
                .windows(2)
                .map(|w| {
                    if w[0].close > 0.0 {
                        (w[1].close - w[0].close) / w[0].close
                    } else {
                        0.0
                    }
                })
                .collect()
        })
    }

    /// Calculate volatility (standard deviation of returns) for a symbol.
    pub fn calculate_volatility(&self, symbol: &str, window: usize) -> Option<Vec<f64>> {
        let returns = self.calculate_returns(symbol)?;

        if returns.len() < window {
            return None;
        }

        let volatilities: Vec<f64> = returns
            .windows(window)
            .map(|w| {
                let mean = w.iter().sum::<f64>() / w.len() as f64;
                let variance = w.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / w.len() as f64;
                variance.sqrt()
            })
            .collect();

        Some(volatilities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ohlcv_typical_price() {
        let candle = OHLCV::new(0, 100.0, 110.0, 95.0, 105.0, 1000.0);
        let expected = (110.0 + 95.0 + 105.0) / 3.0;
        assert!((candle.typical_price() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_ohlcv_return() {
        let candle = OHLCV::new(0, 100.0, 110.0, 95.0, 105.0, 1000.0);
        assert!((candle.return_pct() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_ohlcv_bullish_bearish() {
        let bullish = OHLCV::new(0, 100.0, 110.0, 95.0, 105.0, 1000.0);
        let bearish = OHLCV::new(0, 100.0, 110.0, 95.0, 95.0, 1000.0);

        assert!(bullish.is_bullish());
        assert!(!bullish.is_bearish());
        assert!(bearish.is_bearish());
        assert!(!bearish.is_bullish());
    }

    #[test]
    fn test_multi_symbol_data() {
        let mut data = MultiSymbolData::new();

        let btc_data = vec![
            OHLCV::new(1, 50000.0, 51000.0, 49000.0, 50500.0, 100.0),
            OHLCV::new(2, 50500.0, 52000.0, 50000.0, 51000.0, 120.0),
        ];

        data.add_symbol("BTCUSDT".to_string(), btc_data);

        assert_eq!(data.len(), 1);
        assert!(data.get("BTCUSDT").is_some());
        assert!(data.get("ETHUSDT").is_none());
    }
}
