//! OHLCV (Open, High, Low, Close, Volume) data structures
//!
//! This module provides core data structures for representing
//! candlestick/kline data from cryptocurrency exchanges.

use anyhow::Result;
use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

/// Single OHLCV data point (candlestick)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCVData {
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
    /// Turnover (quote volume)
    pub turnover: f64,
}

impl OHLCVData {
    /// Create a new OHLCV data point
    pub fn new(
        timestamp: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        turnover: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            turnover,
        }
    }

    /// Get datetime from timestamp
    pub fn datetime(&self) -> DateTime<Utc> {
        Utc.timestamp_millis_opt(self.timestamp).unwrap()
    }

    /// Calculate the body size (absolute difference between open and close)
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Calculate the full range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate upper shadow (wick)
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.close.max(self.open)
    }

    /// Calculate lower shadow (wick)
    pub fn lower_shadow(&self) -> f64 {
        self.close.min(self.open) - self.low
    }

    /// Check if this is a bullish (green) candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if this is a bearish (red) candle
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Calculate return from open to close
    pub fn return_pct(&self) -> f64 {
        if self.open != 0.0 {
            (self.close - self.open) / self.open * 100.0
        } else {
            0.0
        }
    }
}

/// Collection of OHLCV data points for a symbol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCVDataset {
    /// Trading symbol (e.g., "BTCUSDT")
    pub symbol: String,
    /// Time interval (e.g., "15" for 15 minutes)
    pub interval: String,
    /// Vector of OHLCV data points
    pub data: Vec<OHLCVData>,
}

impl OHLCVDataset {
    /// Create a new dataset
    pub fn new(symbol: String, interval: String, data: Vec<OHLCVData>) -> Self {
        Self {
            symbol,
            interval,
            data,
        }
    }

    /// Get the number of data points
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get data point by index
    pub fn get(&self, index: usize) -> Option<&OHLCVData> {
        self.data.get(index)
    }

    /// Get closing prices as a vector
    pub fn closes(&self) -> Vec<f64> {
        self.data.iter().map(|d| d.close).collect()
    }

    /// Get volumes as a vector
    pub fn volumes(&self) -> Vec<f64> {
        self.data.iter().map(|d| d.volume).collect()
    }

    /// Get returns as a vector
    pub fn returns(&self) -> Vec<f64> {
        if self.data.len() < 2 {
            return vec![];
        }

        self.data
            .windows(2)
            .map(|w| {
                if w[0].close != 0.0 {
                    (w[1].close - w[0].close) / w[0].close * 100.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Get time range
    pub fn time_range(&self) -> Option<(DateTime<Utc>, DateTime<Utc>)> {
        if self.data.is_empty() {
            return None;
        }
        Some((
            self.data.first().unwrap().datetime(),
            self.data.last().unwrap().datetime(),
        ))
    }

    /// Save dataset to CSV file
    pub fn to_csv<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        writeln!(writer, "timestamp,open,high,low,close,volume,turnover")?;

        // Write data
        for candle in &self.data {
            writeln!(
                writer,
                "{},{},{},{},{},{},{}",
                candle.timestamp,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
                candle.turnover
            )?;
        }

        Ok(())
    }

    /// Load dataset from CSV file
    pub fn from_csv<P: AsRef<Path>>(path: P, symbol: &str, interval: &str) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut csv_reader = csv::Reader::from_reader(reader);

        let mut data = Vec::new();

        for result in csv_reader.deserialize() {
            let record: OHLCVData = result?;
            data.push(record);
        }

        Ok(Self::new(symbol.to_string(), interval.to_string(), data))
    }

    /// Slice dataset by timestamp range
    pub fn slice_by_time(&self, start: i64, end: i64) -> Self {
        let data: Vec<OHLCVData> = self
            .data
            .iter()
            .filter(|d| d.timestamp >= start && d.timestamp <= end)
            .cloned()
            .collect();

        Self::new(self.symbol.clone(), self.interval.clone(), data)
    }

    /// Split dataset into train, validation, and test sets
    pub fn train_val_test_split(
        &self,
        train_ratio: f64,
        val_ratio: f64,
    ) -> (Self, Self, Self) {
        let n = self.data.len();
        let train_end = (n as f64 * train_ratio) as usize;
        let val_end = train_end + (n as f64 * val_ratio) as usize;

        let train_data = self.data[..train_end].to_vec();
        let val_data = self.data[train_end..val_end].to_vec();
        let test_data = self.data[val_end..].to_vec();

        (
            Self::new(self.symbol.clone(), self.interval.clone(), train_data),
            Self::new(self.symbol.clone(), self.interval.clone(), val_data),
            Self::new(self.symbol.clone(), self.interval.clone(), test_data),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ohlcv_data() {
        let candle = OHLCVData::new(
            1704067200000,
            42000.0,
            42500.0,
            41800.0,
            42300.0,
            1000.0,
            42150000.0,
        );

        assert!(candle.is_bullish());
        assert!(!candle.is_bearish());
        assert_eq!(candle.body_size(), 300.0);
        assert_eq!(candle.range(), 700.0);
    }

    #[test]
    fn test_dataset_split() {
        let data: Vec<OHLCVData> = (0..100)
            .map(|i| {
                OHLCVData::new(
                    i as i64 * 60000,
                    100.0 + i as f64,
                    101.0 + i as f64,
                    99.0 + i as f64,
                    100.5 + i as f64,
                    1000.0,
                    100500.0,
                )
            })
            .collect();

        let dataset = OHLCVDataset::new("TEST".to_string(), "1".to_string(), data);
        let (train, val, test) = dataset.train_val_test_split(0.7, 0.15);

        assert_eq!(train.len(), 70);
        assert_eq!(val.len(), 15);
        assert_eq!(test.len(), 15);
    }
}
