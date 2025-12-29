//! OHLCV (Open, High, Low, Close, Volume) data structures
//!
//! Core data structures for representing candlestick/kline data

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turnover: Option<f64>,
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
            turnover: None,
        }
    }

    /// Calculate the range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate the body (close - open)
    pub fn body(&self) -> f64 {
        self.close - self.open
    }

    /// Calculate the typical price (H + L + C) / 3
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate the VWAP-like price
    pub fn vwap(&self) -> f64 {
        if self.volume > 0.0 {
            if let Some(turnover) = self.turnover {
                turnover / self.volume
            } else {
                self.typical_price()
            }
        } else {
            self.typical_price()
        }
    }

    /// Check if candle is bullish (green)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if candle is bearish (red)
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Calculate return from open to close
    pub fn return_pct(&self) -> f64 {
        if self.open > 0.0 {
            (self.close - self.open) / self.open
        } else {
            0.0
        }
    }

    /// Calculate position of close within the range [0, 1]
    pub fn close_position(&self) -> f64 {
        let range = self.range();
        if range > 0.0 {
            (self.close - self.low) / range
        } else {
            0.5
        }
    }
}

/// Series of OHLCV data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCVSeries {
    pub symbol: String,
    pub interval: String,
    pub data: Vec<OHLCV>,
}

impl OHLCVSeries {
    /// Create an empty series
    pub fn new(symbol: String, interval: String) -> Self {
        Self {
            symbol,
            interval,
            data: Vec::new(),
        }
    }

    /// Create a series with data
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

    /// Get the number of candles
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Get all close prices
    pub fn closes(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.close).collect()
    }

    /// Get all open prices
    pub fn opens(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.open).collect()
    }

    /// Get all high prices
    pub fn highs(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.high).collect()
    }

    /// Get all low prices
    pub fn lows(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.low).collect()
    }

    /// Get all volumes
    pub fn volumes(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.volume).collect()
    }

    /// Get all timestamps
    pub fn timestamps(&self) -> Vec<DateTime<Utc>> {
        self.data.iter().map(|c| c.timestamp).collect()
    }

    /// Calculate returns (close-to-close)
    pub fn returns(&self) -> Vec<f64> {
        if self.data.len() < 2 {
            return vec![];
        }

        self.data
            .windows(2)
            .map(|w| {
                if w[0].close > 0.0 {
                    (w[1].close - w[0].close) / w[0].close
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Calculate log returns
    pub fn log_returns(&self) -> Vec<f64> {
        if self.data.len() < 2 {
            return vec![];
        }

        self.data
            .windows(2)
            .map(|w| {
                if w[0].close > 0.0 && w[1].close > 0.0 {
                    (w[1].close / w[0].close).ln()
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Get the latest candle
    pub fn latest(&self) -> Option<&OHLCV> {
        self.data.last()
    }

    /// Get candle at index
    pub fn get(&self, index: usize) -> Option<&OHLCV> {
        self.data.get(index)
    }

    /// Append a new candle
    pub fn push(&mut self, candle: OHLCV) {
        self.data.push(candle);
    }

    /// Slice the series
    pub fn slice(&self, start: usize, end: usize) -> OHLCVSeries {
        let end = end.min(self.data.len());
        let start = start.min(end);

        OHLCVSeries {
            symbol: self.symbol.clone(),
            interval: self.interval.clone(),
            data: self.data[start..end].to_vec(),
        }
    }

    /// Get the last n candles
    pub fn tail(&self, n: usize) -> OHLCVSeries {
        let start = self.data.len().saturating_sub(n);
        self.slice(start, self.data.len())
    }

    /// Save to CSV file
    pub fn to_csv(&self, path: &str) -> anyhow::Result<()> {
        let mut writer = csv::Writer::from_path(path)?;

        // Write header
        writer.write_record([
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
        ])?;

        // Write data
        for candle in &self.data {
            writer.write_record([
                candle.timestamp.to_rfc3339(),
                candle.open.to_string(),
                candle.high.to_string(),
                candle.low.to_string(),
                candle.close.to_string(),
                candle.volume.to_string(),
                candle.turnover.map(|t| t.to_string()).unwrap_or_default(),
            ])?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Load from CSV file
    pub fn from_csv(path: &str, symbol: &str, interval: &str) -> anyhow::Result<Self> {
        let mut reader = csv::Reader::from_path(path)?;
        let mut data = Vec::new();

        for result in reader.records() {
            let record = result?;

            let timestamp: DateTime<Utc> = record[0].parse()?;
            let open: f64 = record[1].parse()?;
            let high: f64 = record[2].parse()?;
            let low: f64 = record[3].parse()?;
            let close: f64 = record[4].parse()?;
            let volume: f64 = record[5].parse()?;
            let turnover: Option<f64> = record.get(6).and_then(|s| s.parse().ok());

            let mut candle = OHLCV::new(timestamp, open, high, low, close, volume);
            candle.turnover = turnover;
            data.push(candle);
        }

        Ok(Self::with_data(symbol.to_string(), interval.to_string(), data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_ohlcv_calculations() {
        let candle = OHLCV::new(
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            100.0,
            110.0,
            90.0,
            105.0,
            1000.0,
        );

        assert_eq!(candle.range(), 20.0);
        assert_eq!(candle.body(), 5.0);
        assert!(candle.is_bullish());
        assert!(!candle.is_bearish());
        assert!((candle.return_pct() - 0.05).abs() < 1e-10);
        assert!((candle.close_position() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_ohlcv_series_returns() {
        let data = vec![
            OHLCV::new(Utc::now(), 100.0, 105.0, 95.0, 100.0, 100.0),
            OHLCV::new(Utc::now(), 100.0, 110.0, 95.0, 110.0, 100.0),
            OHLCV::new(Utc::now(), 110.0, 115.0, 100.0, 105.0, 100.0),
        ];

        let series = OHLCVSeries::with_data("TEST".to_string(), "60".to_string(), data);
        let returns = series.returns();

        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.1).abs() < 1e-10); // 100 -> 110 = 10%
        assert!((returns[1] - (-0.0454545)).abs() < 1e-5); // 110 -> 105 ~ -4.5%
    }
}
