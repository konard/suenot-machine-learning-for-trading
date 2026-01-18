//! OHLCV (Open, High, Low, Close, Volume) data structures
//!
//! Provides core data types for candlestick/kline data

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Single OHLCV candlestick
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCV {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub turnover: Option<f64>,
}

impl OHLCV {
    /// Create new OHLCV candle
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
            turnover: None,
        }
    }

    /// Calculate candle range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate body size (|close - open|)
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Check if bullish candle (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Calculate return from open to close
    pub fn return_pct(&self) -> f64 {
        if self.open == 0.0 {
            0.0
        } else {
            (self.close - self.open) / self.open * 100.0
        }
    }

    /// Calculate typical price (HLC average)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }
}

/// Series of OHLCV data with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCVSeries {
    pub symbol: String,
    pub interval: String,
    pub data: Vec<OHLCV>,
}

impl OHLCVSeries {
    /// Create new empty series
    pub fn new(symbol: String, interval: String) -> Self {
        Self {
            symbol,
            interval,
            data: Vec::new(),
        }
    }

    /// Create series with existing data
    pub fn with_data(symbol: String, interval: String, data: Vec<OHLCV>) -> Self {
        Self {
            symbol,
            interval,
            data,
        }
    }

    /// Check if series is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get series length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Get all close prices
    pub fn closes(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.close).collect()
    }

    /// Get all volumes
    pub fn volumes(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.volume).collect()
    }

    /// Get all high prices
    pub fn highs(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.high).collect()
    }

    /// Get all low prices
    pub fn lows(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.low).collect()
    }

    /// Calculate returns series
    pub fn returns(&self) -> Vec<f64> {
        if self.data.len() < 2 {
            return Vec::new();
        }

        self.data
            .windows(2)
            .map(|w| {
                if w[0].close == 0.0 {
                    0.0
                } else {
                    (w[1].close - w[0].close) / w[0].close * 100.0
                }
            })
            .collect()
    }

    /// Calculate log returns series
    pub fn log_returns(&self) -> Vec<f64> {
        if self.data.len() < 2 {
            return Vec::new();
        }

        self.data
            .windows(2)
            .map(|w| {
                if w[0].close <= 0.0 || w[1].close <= 0.0 {
                    0.0
                } else {
                    (w[1].close / w[0].close).ln()
                }
            })
            .collect()
    }

    /// Get last N candles
    pub fn tail(&self, n: usize) -> Vec<&OHLCV> {
        let start = self.data.len().saturating_sub(n);
        self.data[start..].iter().collect()
    }

    /// Calculate rolling volatility (standard deviation of returns)
    pub fn rolling_volatility(&self, window: usize) -> Vec<f64> {
        let returns = self.returns();
        if returns.len() < window {
            return Vec::new();
        }

        returns
            .windows(window)
            .map(|w| {
                let mean: f64 = w.iter().sum::<f64>() / w.len() as f64;
                let variance: f64 =
                    w.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / w.len() as f64;
                variance.sqrt()
            })
            .collect()
    }

    /// Calculate maximum drawdown
    pub fn max_drawdown(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }

        let mut peak = self.data[0].close;
        let mut max_dd = 0.0;

        for candle in &self.data {
            if candle.close > peak {
                peak = candle.close;
            }
            let dd = (peak - candle.close) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        max_dd * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_ohlcv_creation() {
        let candle = OHLCV::new(Utc::now(), 100.0, 110.0, 95.0, 105.0, 1000.0);
        assert_eq!(candle.range(), 15.0);
        assert_eq!(candle.body(), 5.0);
        assert!(candle.is_bullish());
    }

    #[test]
    fn test_returns_calculation() {
        let data = vec![
            OHLCV::new(Utc::now(), 100.0, 100.0, 100.0, 100.0, 100.0),
            OHLCV::new(Utc::now(), 100.0, 100.0, 100.0, 110.0, 100.0),
            OHLCV::new(Utc::now(), 110.0, 110.0, 110.0, 105.0, 100.0),
        ];
        let series = OHLCVSeries::with_data("TEST".into(), "1h".into(), data);
        let returns = series.returns();

        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 10.0).abs() < 0.01); // 10% return
    }
}
