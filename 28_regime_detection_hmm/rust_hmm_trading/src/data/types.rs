//! Data types for market data

use serde::{Deserialize, Serialize};

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Unix timestamp in milliseconds
    pub timestamp: u64,
    /// Opening price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Trading volume
    pub volume: f64,
    /// Turnover (quote volume)
    pub turnover: f64,
}

impl Candle {
    /// Calculate return from open to close
    pub fn return_oc(&self) -> f64 {
        (self.close - self.open) / self.open
    }

    /// Calculate high-low range
    pub fn range(&self) -> f64 {
        (self.high - self.low) / self.open
    }

    /// Calculate body size (absolute)
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs() / self.open
    }

    /// Check if bullish candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Get typical price (HLC average)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }
}

/// OHLCV type alias
pub type OHLCV = Candle;

/// Dataset containing candles and computed features
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Raw candle data
    pub candles: Vec<Candle>,
    /// Symbol name
    pub symbol: String,
    /// Interval string
    pub interval: String,
}

impl Dataset {
    /// Create new dataset from candles
    pub fn new(candles: Vec<Candle>, symbol: &str, interval: &str) -> Self {
        Self {
            candles,
            symbol: symbol.to_string(),
            interval: interval.to_string(),
        }
    }

    /// Get closing prices
    pub fn closes(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.close).collect()
    }

    /// Get returns
    pub fn returns(&self) -> Vec<f64> {
        let closes = self.closes();
        closes
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    /// Get log returns
    pub fn log_returns(&self) -> Vec<f64> {
        let closes = self.closes();
        closes.windows(2).map(|w| (w[1] / w[0]).ln()).collect()
    }

    /// Get volumes
    pub fn volumes(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.volume).collect()
    }

    /// Get timestamps
    pub fn timestamps(&self) -> Vec<u64> {
        self.candles.iter().map(|c| c.timestamp).collect()
    }

    /// Number of candles
    pub fn len(&self) -> usize {
        self.candles.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.candles.is_empty()
    }

    /// Get slice of candles
    pub fn slice(&self, start: usize, end: usize) -> Dataset {
        Dataset {
            candles: self.candles[start..end].to_vec(),
            symbol: self.symbol.clone(),
            interval: self.interval.clone(),
        }
    }

    /// Split into train/test sets
    pub fn train_test_split(&self, train_ratio: f64) -> (Dataset, Dataset) {
        let split_idx = (self.len() as f64 * train_ratio) as usize;
        (
            self.slice(0, split_idx),
            self.slice(split_idx, self.len()),
        )
    }
}

/// Save dataset to CSV
impl Dataset {
    pub fn to_csv(&self, path: &str) -> anyhow::Result<()> {
        let mut writer = csv::Writer::from_path(path)?;

        writer.write_record(["timestamp", "open", "high", "low", "close", "volume", "turnover"])?;

        for candle in &self.candles {
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

    pub fn from_csv(path: &str, symbol: &str, interval: &str) -> anyhow::Result<Self> {
        let mut reader = csv::Reader::from_path(path)?;
        let mut candles = Vec::new();

        for result in reader.records() {
            let record = result?;
            candles.push(Candle {
                timestamp: record[0].parse()?,
                open: record[1].parse()?,
                high: record[2].parse()?,
                low: record[3].parse()?,
                close: record[4].parse()?,
                volume: record[5].parse()?,
                turnover: record[6].parse()?,
            });
        }

        Ok(Self::new(candles, symbol, interval))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_candle() -> Candle {
        Candle {
            timestamp: 1000000,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        }
    }

    #[test]
    fn test_candle_return() {
        let candle = sample_candle();
        assert!((candle.return_oc() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_candle_range() {
        let candle = sample_candle();
        assert!((candle.range() - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_candle_is_bullish() {
        let candle = sample_candle();
        assert!(candle.is_bullish());
    }
}
