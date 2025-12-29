//! OHLCV Data Structures
//!
//! Standard candlestick data representation

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Single OHLCV candle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCV {
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
    /// Turnover (optional)
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

    /// Returns true if this is a bullish candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Returns true if this is a bearish candle
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Calculate the body size (absolute difference between open and close)
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Calculate the upper shadow size
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.close.max(self.open)
    }

    /// Calculate the lower shadow size
    pub fn lower_shadow(&self) -> f64 {
        self.close.min(self.open) - self.low
    }

    /// Calculate the range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate typical price (HLC/3)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate return from open to close
    pub fn return_pct(&self) -> f64 {
        if self.open > 0.0 {
            (self.close - self.open) / self.open
        } else {
            0.0
        }
    }
}

/// Time series of OHLCV data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCVSeries {
    pub symbol: String,
    pub interval: String,
    pub data: Vec<OHLCV>,
}

impl OHLCVSeries {
    pub fn new(symbol: String, interval: String) -> Self {
        Self {
            symbol,
            interval,
            data: Vec::new(),
        }
    }

    pub fn with_data(symbol: String, interval: String, data: Vec<OHLCV>) -> Self {
        Self {
            symbol,
            interval,
            data,
        }
    }

    /// Get number of candles
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get closing prices as vector
    pub fn close_prices(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.close).collect()
    }

    /// Get opening prices as vector
    pub fn open_prices(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.open).collect()
    }

    /// Get high prices as vector
    pub fn high_prices(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.high).collect()
    }

    /// Get low prices as vector
    pub fn low_prices(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.low).collect()
    }

    /// Get volumes as vector
    pub fn volumes(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.volume).collect()
    }

    /// Get returns as vector
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

    /// Get log returns as vector
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

    /// Sort by timestamp ascending
    pub fn sort_by_time(&mut self) {
        self.data.sort_by_key(|c| c.timestamp);
    }

    /// Get slice of data
    pub fn slice(&self, start: usize, end: usize) -> OHLCVSeries {
        let end = end.min(self.data.len());
        let start = start.min(end);
        OHLCVSeries::with_data(
            self.symbol.clone(),
            self.interval.clone(),
            self.data[start..end].to_vec(),
        )
    }

    /// Save to CSV
    pub fn save_csv(&self, path: &str) -> anyhow::Result<()> {
        let mut writer = csv::Writer::from_path(path)?;

        writer.write_record(&["timestamp", "open", "high", "low", "close", "volume"])?;

        for candle in &self.data {
            writer.write_record(&[
                candle.timestamp.to_rfc3339(),
                candle.open.to_string(),
                candle.high.to_string(),
                candle.low.to_string(),
                candle.close.to_string(),
                candle.volume.to_string(),
            ])?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Load from CSV
    pub fn load_csv(path: &str, symbol: String, interval: String) -> anyhow::Result<Self> {
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

            data.push(OHLCV::new(timestamp, open, high, low, close, volume));
        }

        Ok(Self::with_data(symbol, interval, data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_ohlcv_bullish() {
        let candle = OHLCV::new(
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            100.0,
            110.0,
            95.0,
            105.0,
            1000.0,
        );
        assert!(candle.is_bullish());
        assert!(!candle.is_bearish());
    }

    #[test]
    fn test_returns() {
        let data = vec![
            OHLCV::new(
                Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
                100.0, 110.0, 95.0, 100.0, 1000.0,
            ),
            OHLCV::new(
                Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
                100.0, 115.0, 98.0, 110.0, 1200.0,
            ),
        ];

        let series = OHLCVSeries::with_data("BTCUSDT".to_string(), "1h".to_string(), data);
        let returns = series.returns();

        assert_eq!(returns.len(), 1);
        assert!((returns[0] - 0.1).abs() < 1e-10);
    }
}
