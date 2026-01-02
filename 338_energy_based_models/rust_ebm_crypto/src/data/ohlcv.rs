//! OHLCV data structures for market data representation

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Single OHLCV candle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Timestamp of the candle
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
    #[serde(default)]
    pub turnover: f64,
}

impl Candle {
    /// Create a new candle
    pub fn new(
        timestamp: i64,
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
            turnover: 0.0,
        }
    }

    /// Get the candle range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Get the candle body (close - open)
    pub fn body(&self) -> f64 {
        self.close - self.open
    }

    /// Check if candle is bullish (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Get the upper wick size
    pub fn upper_wick(&self) -> f64 {
        if self.is_bullish() {
            self.high - self.close
        } else {
            self.high - self.open
        }
    }

    /// Get the lower wick size
    pub fn lower_wick(&self) -> f64 {
        if self.is_bullish() {
            self.open - self.low
        } else {
            self.close - self.low
        }
    }

    /// Get the close position within the range (0 = low, 1 = high)
    pub fn close_position(&self) -> f64 {
        let range = self.range();
        if range == 0.0 {
            0.5
        } else {
            (self.close - self.low) / range
        }
    }

    /// Get datetime from timestamp
    pub fn datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp(self.timestamp / 1000, 0)
            .unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap())
    }
}

/// Collection of OHLCV candles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OhlcvData {
    /// Symbol name (e.g., "BTCUSDT")
    pub symbol: String,
    /// Timeframe (e.g., "1", "60", "D")
    pub interval: String,
    /// Vector of candles
    pub data: Vec<Candle>,
}

impl OhlcvData {
    /// Create new OHLCV data collection
    pub fn new(symbol: &str, interval: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            interval: interval.to_string(),
            data: Vec::new(),
        }
    }

    /// Add a candle to the collection
    pub fn push(&mut self, candle: Candle) {
        self.data.push(candle);
    }

    /// Get the number of candles
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get closing prices as a vector
    pub fn closes(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.close).collect()
    }

    /// Get opening prices as a vector
    pub fn opens(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.open).collect()
    }

    /// Get high prices as a vector
    pub fn highs(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.high).collect()
    }

    /// Get low prices as a vector
    pub fn lows(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.low).collect()
    }

    /// Get volumes as a vector
    pub fn volumes(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.volume).collect()
    }

    /// Get timestamps as a vector
    pub fn timestamps(&self) -> Vec<i64> {
        self.data.iter().map(|c| c.timestamp).collect()
    }

    /// Calculate returns (percentage change)
    pub fn returns(&self) -> Vec<f64> {
        if self.data.len() < 2 {
            return vec![];
        }

        self.data
            .windows(2)
            .map(|w| (w[1].close - w[0].close) / w[0].close)
            .collect()
    }

    /// Calculate log returns
    pub fn log_returns(&self) -> Vec<f64> {
        if self.data.len() < 2 {
            return vec![];
        }

        self.data
            .windows(2)
            .map(|w| (w[1].close / w[0].close).ln())
            .collect()
    }

    /// Sort candles by timestamp (ascending)
    pub fn sort_by_time(&mut self) {
        self.data.sort_by_key(|c| c.timestamp);
    }

    /// Get slice of data
    pub fn slice(&self, start: usize, end: usize) -> OhlcvData {
        let end = end.min(self.data.len());
        let start = start.min(end);

        OhlcvData {
            symbol: self.symbol.clone(),
            interval: self.interval.clone(),
            data: self.data[start..end].to_vec(),
        }
    }

    /// Save to CSV file
    pub fn to_csv(&self, path: &str) -> anyhow::Result<()> {
        let mut writer = csv::Writer::from_path(path)?;

        writer.write_record(&["timestamp", "open", "high", "low", "close", "volume", "turnover"])?;

        for candle in &self.data {
            writer.write_record(&[
                candle.timestamp.to_string(),
                candle.open.to_string(),
                candle.high.to_string(),
                candle.low.to_string(),
                candle.close.to_string(),
                candle.volume.to_string(),
                candle.turnover.to_string(),
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
            let candle = Candle {
                timestamp: record[0].parse()?,
                open: record[1].parse()?,
                high: record[2].parse()?,
                low: record[3].parse()?,
                close: record[4].parse()?,
                volume: record[5].parse()?,
                turnover: record.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.0),
            };
            data.push(candle);
        }

        Ok(OhlcvData {
            symbol: symbol.to_string(),
            interval: interval.to_string(),
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_properties() {
        let candle = Candle::new(1000000, 100.0, 110.0, 95.0, 105.0, 1000.0);

        assert_eq!(candle.range(), 15.0);
        assert_eq!(candle.body(), 5.0);
        assert!(candle.is_bullish());
        assert_eq!(candle.upper_wick(), 5.0);
        assert_eq!(candle.lower_wick(), 5.0);
    }

    #[test]
    fn test_ohlcv_returns() {
        let mut ohlcv = OhlcvData::new("TEST", "1");
        ohlcv.push(Candle::new(1000, 100.0, 110.0, 90.0, 100.0, 1000.0));
        ohlcv.push(Candle::new(2000, 100.0, 120.0, 95.0, 110.0, 1200.0));
        ohlcv.push(Candle::new(3000, 110.0, 115.0, 100.0, 105.0, 800.0));

        let returns = ohlcv.returns();
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.1).abs() < 1e-10); // 10% return
    }
}
