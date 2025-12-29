//! OHLCV data structures.

use serde::{Deserialize, Serialize};

/// OHLCV (Open, High, Low, Close, Volume) candlestick data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCV {
    /// Timestamp in milliseconds
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
}

impl OHLCV {
    /// Create a new OHLCV candle.
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

    /// Calculate the typical price (HLC/3).
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate the range (high - low).
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate the body (close - open).
    pub fn body(&self) -> f64 {
        self.close - self.open
    }

    /// Check if the candle is bullish (close > open).
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Calculate return from open to close.
    pub fn return_pct(&self) -> f64 {
        if self.open == 0.0 {
            0.0
        } else {
            (self.close - self.open) / self.open
        }
    }
}

/// Multi-symbol OHLCV data collection.
#[derive(Debug, Clone)]
pub struct MultiSymbolData {
    /// Symbol names
    pub symbols: Vec<String>,
    /// OHLCV data per symbol
    pub data: Vec<Vec<OHLCV>>,
}

impl MultiSymbolData {
    /// Create new multi-symbol data collection.
    pub fn new() -> Self {
        Self {
            symbols: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Add symbol data.
    pub fn add_symbol(&mut self, symbol: String, ohlcv: Vec<OHLCV>) {
        self.symbols.push(symbol);
        self.data.push(ohlcv);
    }

    /// Get number of symbols.
    pub fn num_symbols(&self) -> usize {
        self.symbols.len()
    }

    /// Get data for a specific symbol.
    pub fn get_symbol(&self, symbol: &str) -> Option<&Vec<OHLCV>> {
        self.symbols
            .iter()
            .position(|s| s == symbol)
            .map(|idx| &self.data[idx])
    }

    /// Calculate returns for all symbols.
    pub fn compute_returns(&self) -> Vec<Vec<f64>> {
        self.data
            .iter()
            .map(|ohlcv_vec| {
                ohlcv_vec
                    .windows(2)
                    .map(|w| {
                        if w[0].close == 0.0 {
                            0.0
                        } else {
                            (w[1].close - w[0].close) / w[0].close
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Align timestamps across all symbols (find common timestamps).
    pub fn align_timestamps(&self) -> Option<Vec<i64>> {
        if self.data.is_empty() {
            return None;
        }

        // Get timestamps from first symbol
        let first_timestamps: std::collections::HashSet<i64> =
            self.data[0].iter().map(|o| o.timestamp).collect();

        // Find intersection with all other symbols
        let common: std::collections::HashSet<i64> = self.data[1..]
            .iter()
            .fold(first_timestamps, |acc, ohlcv_vec| {
                let timestamps: std::collections::HashSet<i64> =
                    ohlcv_vec.iter().map(|o| o.timestamp).collect();
                acc.intersection(&timestamps).cloned().collect()
            });

        let mut timestamps: Vec<i64> = common.into_iter().collect();
        timestamps.sort();
        Some(timestamps)
    }
}

impl Default for MultiSymbolData {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ohlcv_calculations() {
        let candle = OHLCV::new(0, 100.0, 110.0, 95.0, 105.0, 1000.0);

        assert!((candle.typical_price() - 103.333).abs() < 0.01);
        assert!((candle.range() - 15.0).abs() < 0.001);
        assert!((candle.body() - 5.0).abs() < 0.001);
        assert!(candle.is_bullish());
        assert!((candle.return_pct() - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_multi_symbol_data() {
        let mut data = MultiSymbolData::new();

        data.add_symbol(
            "BTC".to_string(),
            vec![
                OHLCV::new(1, 100.0, 110.0, 95.0, 105.0, 1000.0),
                OHLCV::new(2, 105.0, 115.0, 100.0, 110.0, 1100.0),
            ],
        );

        assert_eq!(data.num_symbols(), 1);
        assert!(data.get_symbol("BTC").is_some());
        assert!(data.get_symbol("ETH").is_none());
    }
}
