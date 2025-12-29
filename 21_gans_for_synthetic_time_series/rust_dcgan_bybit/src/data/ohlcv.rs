//! OHLCV (Open, High, Low, Close, Volume) data structures

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Single OHLCV candlestick data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCVData {
    /// Unix timestamp in milliseconds
    pub timestamp: i64,
    /// Opening price
    pub open: f64,
    /// Highest price during the period
    pub high: f64,
    /// Lowest price during the period
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
        DateTime::from_timestamp_millis(self.timestamp)
            .unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap())
    }

    /// Convert to feature vector [open, high, low, close, volume]
    pub fn to_features(&self) -> [f64; 5] {
        [self.open, self.high, self.low, self.close, self.volume]
    }

    /// Calculate returns (percentage change in close price)
    pub fn returns(&self, previous: &OHLCVData) -> f64 {
        if previous.close == 0.0 {
            0.0
        } else {
            (self.close - previous.close) / previous.close
        }
    }

    /// Calculate log returns
    pub fn log_returns(&self, previous: &OHLCVData) -> f64 {
        if previous.close <= 0.0 || self.close <= 0.0 {
            0.0
        } else {
            (self.close / previous.close).ln()
        }
    }
}

/// Collection of OHLCV data with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCVDataset {
    /// Trading symbol (e.g., "BTCUSDT")
    pub symbol: String,
    /// Timeframe interval (e.g., "1h", "4h", "1d")
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

    /// Get number of data points
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Calculate log returns for the entire dataset
    pub fn calculate_log_returns(&self) -> Vec<f64> {
        if self.data.len() < 2 {
            return vec![];
        }

        self.data
            .windows(2)
            .map(|w| w[1].log_returns(&w[0]))
            .collect()
    }

    /// Get close prices
    pub fn close_prices(&self) -> Vec<f64> {
        self.data.iter().map(|d| d.close).collect()
    }

    /// Get all features as a 2D vector
    pub fn to_feature_matrix(&self) -> Vec<[f64; 5]> {
        self.data.iter().map(|d| d.to_features()).collect()
    }

    /// Save dataset to CSV file
    pub fn save_csv(&self, path: &str) -> anyhow::Result<()> {
        let mut writer = csv::Writer::from_path(path)?;

        writer.write_record([
            "timestamp", "open", "high", "low", "close", "volume", "turnover",
        ])?;

        for record in &self.data {
            writer.write_record([
                record.timestamp.to_string(),
                record.open.to_string(),
                record.high.to_string(),
                record.low.to_string(),
                record.close.to_string(),
                record.volume.to_string(),
                record.turnover.to_string(),
            ])?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Load dataset from CSV file
    pub fn load_csv(path: &str, symbol: String, interval: String) -> anyhow::Result<Self> {
        let mut reader = csv::Reader::from_path(path)?;
        let mut data = Vec::new();

        for result in reader.records() {
            let record = result?;
            let ohlcv = OHLCVData::new(
                record[0].parse()?,
                record[1].parse()?,
                record[2].parse()?,
                record[3].parse()?,
                record[4].parse()?,
                record[5].parse()?,
                record[6].parse()?,
            );
            data.push(ohlcv);
        }

        Ok(Self::new(symbol, interval, data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ohlcv_returns() {
        let prev = OHLCVData::new(0, 100.0, 105.0, 95.0, 100.0, 1000.0, 100000.0);
        let curr = OHLCVData::new(1, 100.0, 110.0, 98.0, 105.0, 1200.0, 120000.0);

        let returns = curr.returns(&prev);
        assert!((returns - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_log_returns() {
        let prev = OHLCVData::new(0, 100.0, 105.0, 95.0, 100.0, 1000.0, 100000.0);
        let curr = OHLCVData::new(1, 100.0, 110.0, 98.0, 105.0, 1200.0, 120000.0);

        let log_ret = curr.log_returns(&prev);
        let expected = (105.0_f64 / 100.0).ln();
        assert!((log_ret - expected).abs() < 1e-10);
    }
}
