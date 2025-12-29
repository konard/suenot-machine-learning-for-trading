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
    /// Create a new OHLCV record.
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

    /// Calculate the typical price (HLC average).
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate the true range.
    pub fn true_range(&self, prev_close: f64) -> f64 {
        let hl = self.high - self.low;
        let hc = (self.high - prev_close).abs();
        let lc = (self.low - prev_close).abs();
        hl.max(hc).max(lc)
    }

    /// Calculate returns from previous close.
    pub fn returns(&self, prev_close: f64) -> f64 {
        (self.close - prev_close) / prev_close
    }

    /// Calculate log returns from previous close.
    pub fn log_returns(&self, prev_close: f64) -> f64 {
        (self.close / prev_close).ln()
    }
}

/// A collection of OHLCV data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCVDataset {
    pub data: Vec<OHLCV>,
    pub symbol: String,
    pub interval: String,
}

impl OHLCVDataset {
    /// Create a new dataset.
    pub fn new(data: Vec<OHLCV>, symbol: String, interval: String) -> Self {
        Self { data, symbol, interval }
    }

    /// Get the number of records.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get closing prices.
    pub fn closes(&self) -> Vec<f64> {
        self.data.iter().map(|x| x.close).collect()
    }

    /// Get volumes.
    pub fn volumes(&self) -> Vec<f64> {
        self.data.iter().map(|x| x.volume).collect()
    }

    /// Calculate returns.
    pub fn returns(&self) -> Vec<f64> {
        let closes = self.closes();
        closes
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    /// Save to CSV file.
    pub fn to_csv(&self, path: &str) -> anyhow::Result<()> {
        let mut writer = csv::Writer::from_path(path)?;

        writer.write_record(&["timestamp", "open", "high", "low", "close", "volume"])?;

        for record in &self.data {
            writer.write_record(&[
                record.timestamp.to_string(),
                record.open.to_string(),
                record.high.to_string(),
                record.low.to_string(),
                record.close.to_string(),
                record.volume.to_string(),
            ])?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Load from CSV file.
    pub fn from_csv(path: &str, symbol: String, interval: String) -> anyhow::Result<Self> {
        let mut reader = csv::Reader::from_path(path)?;
        let mut data = Vec::new();

        for result in reader.records() {
            let record = result?;
            data.push(OHLCV {
                timestamp: record[0].parse()?,
                open: record[1].parse()?,
                high: record[2].parse()?,
                low: record[3].parse()?,
                close: record[4].parse()?,
                volume: record[5].parse()?,
            });
        }

        Ok(Self::new(data, symbol, interval))
    }
}
