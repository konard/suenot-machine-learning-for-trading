//! OHLCV (Open-High-Low-Close-Volume) data structures

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

    /// Calculate the return for this candle
    pub fn candle_return(&self) -> f64 {
        (self.close - self.open) / self.open
    }

    /// Calculate the range (high-low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate the range relative to close price
    pub fn relative_range(&self) -> f64 {
        self.range() / self.close
    }

    /// Calculate where close is within the candle range (0-1)
    pub fn close_position(&self) -> f64 {
        if self.range() == 0.0 {
            return 0.5;
        }
        (self.close - self.low) / self.range()
    }

    /// Check if this is a bullish candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if this is a bearish candle
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Calculate True Range
    pub fn true_range(&self, prev_close: f64) -> f64 {
        let hl = self.high - self.low;
        let hc = (self.high - prev_close).abs();
        let lc = (self.low - prev_close).abs();
        hl.max(hc).max(lc)
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

    /// Create series with data
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

    /// Get close prices as a vector
    pub fn closes(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.close).collect()
    }

    /// Get volumes as a vector
    pub fn volumes(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.volume).collect()
    }

    /// Calculate returns
    pub fn returns(&self) -> Vec<f64> {
        let closes = self.closes();
        if closes.len() < 2 {
            return Vec::new();
        }

        closes
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    /// Calculate log returns
    pub fn log_returns(&self) -> Vec<f64> {
        let closes = self.closes();
        if closes.len() < 2 {
            return Vec::new();
        }

        closes
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect()
    }

    /// Get a slice of the series
    pub fn slice(&self, start: usize, end: usize) -> OHLCVSeries {
        OHLCVSeries {
            symbol: self.symbol.clone(),
            interval: self.interval.clone(),
            data: self.data[start..end.min(self.len())].to_vec(),
        }
    }

    /// Get the last N candles
    pub fn tail(&self, n: usize) -> OHLCVSeries {
        let start = if self.len() > n { self.len() - n } else { 0 };
        self.slice(start, self.len())
    }

    /// Calculate rolling mean of closes
    pub fn rolling_mean(&self, window: usize) -> Vec<f64> {
        let closes = self.closes();
        if closes.len() < window {
            return Vec::new();
        }

        closes
            .windows(window)
            .map(|w| w.iter().sum::<f64>() / window as f64)
            .collect()
    }

    /// Calculate rolling standard deviation of returns
    pub fn rolling_volatility(&self, window: usize) -> Vec<f64> {
        let returns = self.returns();
        if returns.len() < window {
            return Vec::new();
        }

        returns
            .windows(window)
            .map(|w| {
                let mean = w.iter().sum::<f64>() / window as f64;
                let variance = w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
                variance.sqrt()
            })
            .collect()
    }

    /// Calculate Average True Range
    pub fn atr(&self, window: usize) -> Vec<f64> {
        if self.len() < window + 1 {
            return Vec::new();
        }

        let mut trs: Vec<f64> = Vec::with_capacity(self.len() - 1);
        for i in 1..self.len() {
            trs.push(self.data[i].true_range(self.data[i - 1].close));
        }

        trs.windows(window)
            .map(|w| w.iter().sum::<f64>() / window as f64)
            .collect()
    }

    /// Save to CSV file
    pub fn to_csv(&self, path: &str) -> anyhow::Result<()> {
        let mut wtr = csv::Writer::from_path(path)?;

        wtr.write_record(&[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
        ])?;

        for candle in &self.data {
            wtr.write_record(&[
                candle.timestamp.to_rfc3339(),
                candle.open.to_string(),
                candle.high.to_string(),
                candle.low.to_string(),
                candle.close.to_string(),
                candle.volume.to_string(),
                candle.turnover.map_or("".to_string(), |t| t.to_string()),
            ])?;
        }

        wtr.flush()?;
        Ok(())
    }

    /// Load from CSV file
    pub fn from_csv(path: &str, symbol: &str, interval: &str) -> anyhow::Result<Self> {
        let mut rdr = csv::Reader::from_path(path)?;
        let mut data = Vec::new();

        for result in rdr.records() {
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

    #[test]
    fn test_ohlcv_return() {
        let candle = OHLCV::new(Utc::now(), 100.0, 110.0, 95.0, 105.0, 1000.0);
        assert!((candle.candle_return() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_close_position() {
        let candle = OHLCV::new(Utc::now(), 100.0, 110.0, 90.0, 100.0, 1000.0);
        assert!((candle.close_position() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_series_returns() {
        let series = OHLCVSeries::with_data(
            "TEST".to_string(),
            "1h".to_string(),
            vec![
                OHLCV::new(Utc::now(), 100.0, 105.0, 95.0, 100.0, 1000.0),
                OHLCV::new(Utc::now(), 100.0, 110.0, 98.0, 105.0, 1200.0),
                OHLCV::new(Utc::now(), 105.0, 108.0, 102.0, 103.0, 800.0),
            ],
        );

        let returns = series.returns();
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.05).abs() < 1e-10);
    }
}
