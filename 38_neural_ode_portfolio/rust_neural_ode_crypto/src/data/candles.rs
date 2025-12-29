//! # Candlestick Data Structures
//!
//! This module provides data structures for working with OHLCV candlestick data.

use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};

/// Single candlestick (OHLCV) data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Open timestamp in milliseconds
    pub open_time: i64,
    /// Open price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Close price
    pub close: f64,
    /// Volume (base asset)
    pub volume: f64,
    /// Turnover (quote asset volume)
    pub turnover: f64,
}

impl Candle {
    /// Create a new Candle
    pub fn new(
        open_time: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        turnover: f64,
    ) -> Self {
        Self {
            open_time,
            open,
            high,
            low,
            close,
            volume,
            turnover,
        }
    }

    /// Parse candle from Bybit kline array format
    /// Format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
    pub fn from_bybit_kline(data: &[String]) -> Result<Self> {
        if data.len() < 7 {
            anyhow::bail!("Invalid kline data: expected 7 elements, got {}", data.len());
        }

        Ok(Self {
            open_time: data[0].parse().context("Failed to parse open_time")?,
            open: data[1].parse().context("Failed to parse open")?,
            high: data[2].parse().context("Failed to parse high")?,
            low: data[3].parse().context("Failed to parse low")?,
            close: data[4].parse().context("Failed to parse close")?,
            volume: data[5].parse().context("Failed to parse volume")?,
            turnover: data[6].parse().context("Failed to parse turnover")?,
        })
    }

    /// Get open time as DateTime
    pub fn open_datetime(&self) -> DateTime<Utc> {
        Utc.timestamp_millis_opt(self.open_time).unwrap()
    }

    /// Calculate the return (percentage change)
    pub fn return_pct(&self) -> f64 {
        if self.open == 0.0 {
            0.0
        } else {
            (self.close - self.open) / self.open
        }
    }

    /// Calculate the range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate the range as percentage of open
    pub fn range_pct(&self) -> f64 {
        if self.open == 0.0 {
            0.0
        } else {
            self.range() / self.open
        }
    }

    /// Check if this is a bullish candle (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if this is a bearish candle (close < open)
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Calculate body size (absolute difference between open and close)
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Calculate upper shadow size
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.open.max(self.close)
    }

    /// Calculate lower shadow size
    pub fn lower_shadow(&self) -> f64 {
        self.open.min(self.close) - self.low
    }
}

/// Collection of candles with utility methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleData {
    pub symbol: String,
    pub timeframe: Timeframe,
    pub candles: Vec<Candle>,
}

impl CandleData {
    /// Create new CandleData
    pub fn new(symbol: String, timeframe: Timeframe, candles: Vec<Candle>) -> Self {
        Self {
            symbol,
            timeframe,
            candles,
        }
    }

    /// Get number of candles
    pub fn len(&self) -> usize {
        self.candles.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.candles.is_empty()
    }

    /// Get closing prices as vector
    pub fn close_prices(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.close).collect()
    }

    /// Get returns (percentage changes)
    pub fn returns(&self) -> Vec<f64> {
        self.candles
            .windows(2)
            .map(|w| {
                if w[0].close == 0.0 {
                    0.0
                } else {
                    (w[1].close - w[0].close) / w[0].close
                }
            })
            .collect()
    }

    /// Get log returns
    pub fn log_returns(&self) -> Vec<f64> {
        self.candles
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

    /// Get volumes as vector
    pub fn volumes(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.volume).collect()
    }

    /// Get OHLCV as matrix (n_candles x 5)
    pub fn to_ohlcv_matrix(&self) -> ndarray::Array2<f64> {
        let n = self.candles.len();
        let mut matrix = ndarray::Array2::zeros((n, 5));

        for (i, candle) in self.candles.iter().enumerate() {
            matrix[[i, 0]] = candle.open;
            matrix[[i, 1]] = candle.high;
            matrix[[i, 2]] = candle.low;
            matrix[[i, 3]] = candle.close;
            matrix[[i, 4]] = candle.volume;
        }

        matrix
    }

    /// Get the latest candle
    pub fn latest(&self) -> Option<&Candle> {
        self.candles.last()
    }

    /// Slice candles by index range
    pub fn slice(&self, start: usize, end: usize) -> Self {
        Self {
            symbol: self.symbol.clone(),
            timeframe: self.timeframe.clone(),
            candles: self.candles[start..end.min(self.len())].to_vec(),
        }
    }

    /// Calculate volatility (standard deviation of returns)
    pub fn volatility(&self) -> f64 {
        let returns = self.returns();
        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / returns.len() as f64;

        variance.sqrt()
    }
}

/// Supported timeframes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Timeframe {
    #[serde(rename = "1")]
    Min1,
    #[serde(rename = "3")]
    Min3,
    #[serde(rename = "5")]
    Min5,
    #[serde(rename = "15")]
    Min15,
    #[serde(rename = "30")]
    Min30,
    #[serde(rename = "60")]
    Hour1,
    #[serde(rename = "120")]
    Hour2,
    #[serde(rename = "240")]
    Hour4,
    #[serde(rename = "360")]
    Hour6,
    #[serde(rename = "720")]
    Hour12,
    #[serde(rename = "D")]
    Day,
    #[serde(rename = "W")]
    Week,
    #[serde(rename = "M")]
    Month,
}

impl Timeframe {
    /// Get Bybit API interval string
    pub fn as_str(&self) -> &'static str {
        match self {
            Timeframe::Min1 => "1",
            Timeframe::Min3 => "3",
            Timeframe::Min5 => "5",
            Timeframe::Min15 => "15",
            Timeframe::Min30 => "30",
            Timeframe::Hour1 => "60",
            Timeframe::Hour2 => "120",
            Timeframe::Hour4 => "240",
            Timeframe::Hour6 => "360",
            Timeframe::Hour12 => "720",
            Timeframe::Day => "D",
            Timeframe::Week => "W",
            Timeframe::Month => "M",
        }
    }

    /// Get duration in milliseconds
    pub fn duration_ms(&self) -> i64 {
        match self {
            Timeframe::Min1 => 60_000,
            Timeframe::Min3 => 3 * 60_000,
            Timeframe::Min5 => 5 * 60_000,
            Timeframe::Min15 => 15 * 60_000,
            Timeframe::Min30 => 30 * 60_000,
            Timeframe::Hour1 => 60 * 60_000,
            Timeframe::Hour2 => 2 * 60 * 60_000,
            Timeframe::Hour4 => 4 * 60 * 60_000,
            Timeframe::Hour6 => 6 * 60 * 60_000,
            Timeframe::Hour12 => 12 * 60 * 60_000,
            Timeframe::Day => 24 * 60 * 60_000,
            Timeframe::Week => 7 * 24 * 60 * 60_000,
            Timeframe::Month => 30 * 24 * 60 * 60_000, // Approximate
        }
    }
}

impl Default for Timeframe {
    fn default() -> Self {
        Timeframe::Hour1
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_from_bybit() {
        let data = vec![
            "1704067200000".to_string(),  // timestamp
            "42000.0".to_string(),        // open
            "42500.0".to_string(),        // high
            "41800.0".to_string(),        // low
            "42300.0".to_string(),        // close
            "100.5".to_string(),          // volume
            "4230000.0".to_string(),      // turnover
        ];

        let candle = Candle::from_bybit_kline(&data).unwrap();
        assert_eq!(candle.open, 42000.0);
        assert_eq!(candle.close, 42300.0);
        assert!(candle.is_bullish());
    }

    #[test]
    fn test_candle_calculations() {
        let candle = Candle::new(
            1704067200000,
            100.0, // open
            110.0, // high
            95.0,  // low
            105.0, // close
            1000.0,
            100000.0,
        );

        assert!((candle.return_pct() - 0.05).abs() < 1e-10);
        assert_eq!(candle.range(), 15.0);
        assert!(candle.is_bullish());
        assert!(!candle.is_bearish());
    }

    #[test]
    fn test_candle_data_returns() {
        let candles = vec![
            Candle::new(0, 100.0, 100.0, 100.0, 100.0, 0.0, 0.0),
            Candle::new(1, 100.0, 100.0, 100.0, 110.0, 0.0, 0.0),
            Candle::new(2, 110.0, 110.0, 110.0, 100.0, 0.0, 0.0),
        ];

        let data = CandleData::new("TEST".to_string(), Timeframe::Hour1, candles);
        let returns = data.returns();

        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.1).abs() < 1e-10);  // 10% gain
        assert!((returns[1] - (-0.0909)).abs() < 0.001);  // ~9% loss
    }

    #[test]
    fn test_timeframe_duration() {
        assert_eq!(Timeframe::Min1.duration_ms(), 60_000);
        assert_eq!(Timeframe::Hour1.duration_ms(), 3_600_000);
        assert_eq!(Timeframe::Day.duration_ms(), 86_400_000);
    }
}
