//! Candlestick/OHLCV data structures
//!
//! Provides candle representation and utilities for time series data.

use chrono::{DateTime, Duration, Utc};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Timeframe for candlestick data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Timeframe {
    /// 1 minute
    M1,
    /// 3 minutes
    M3,
    /// 5 minutes
    M5,
    /// 15 minutes
    M15,
    /// 30 minutes
    M30,
    /// 1 hour
    H1,
    /// 2 hours
    H2,
    /// 4 hours
    H4,
    /// 6 hours
    H6,
    /// 12 hours
    H12,
    /// 1 day
    D1,
    /// 1 week
    W1,
    /// 1 month
    MN,
}

impl Timeframe {
    /// Get duration in seconds
    pub fn as_seconds(&self) -> i64 {
        match self {
            Timeframe::M1 => 60,
            Timeframe::M3 => 180,
            Timeframe::M5 => 300,
            Timeframe::M15 => 900,
            Timeframe::M30 => 1800,
            Timeframe::H1 => 3600,
            Timeframe::H2 => 7200,
            Timeframe::H4 => 14400,
            Timeframe::H6 => 21600,
            Timeframe::H12 => 43200,
            Timeframe::D1 => 86400,
            Timeframe::W1 => 604800,
            Timeframe::MN => 2592000, // Approximate
        }
    }

    /// Get duration
    pub fn as_duration(&self) -> Duration {
        Duration::seconds(self.as_seconds())
    }

    /// Convert from Bybit interval string
    pub fn from_bybit_interval(interval: &str) -> Option<Self> {
        match interval {
            "1" => Some(Timeframe::M1),
            "3" => Some(Timeframe::M3),
            "5" => Some(Timeframe::M5),
            "15" => Some(Timeframe::M15),
            "30" => Some(Timeframe::M30),
            "60" => Some(Timeframe::H1),
            "120" => Some(Timeframe::H2),
            "240" => Some(Timeframe::H4),
            "360" => Some(Timeframe::H6),
            "720" => Some(Timeframe::H12),
            "D" => Some(Timeframe::D1),
            "W" => Some(Timeframe::W1),
            "M" => Some(Timeframe::MN),
            _ => None,
        }
    }

    /// Convert to Bybit interval string
    pub fn to_bybit_interval(&self) -> &'static str {
        match self {
            Timeframe::M1 => "1",
            Timeframe::M3 => "3",
            Timeframe::M5 => "5",
            Timeframe::M15 => "15",
            Timeframe::M30 => "30",
            Timeframe::H1 => "60",
            Timeframe::H2 => "120",
            Timeframe::H4 => "240",
            Timeframe::H6 => "360",
            Timeframe::H12 => "720",
            Timeframe::D1 => "D",
            Timeframe::W1 => "W",
            Timeframe::MN => "M",
        }
    }
}

/// OHLCV Candlestick
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Trading symbol
    pub symbol: String,
    /// Candle open time
    pub timestamp: DateTime<Utc>,
    /// Timeframe
    pub timeframe: Timeframe,
    /// Open price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Close price
    pub close: f64,
    /// Volume
    pub volume: f64,
}

impl Candle {
    /// Create a new candle
    pub fn new(
        symbol: &str,
        timestamp: DateTime<Utc>,
        timeframe: Timeframe,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            symbol: symbol.to_string(),
            timestamp,
            timeframe,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Calculate candle body (close - open)
    pub fn body(&self) -> f64 {
        self.close - self.open
    }

    /// Calculate absolute body size
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Calculate candle range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Calculate upper wick
    pub fn upper_wick(&self) -> f64 {
        self.high - self.close.max(self.open)
    }

    /// Calculate lower wick
    pub fn lower_wick(&self) -> f64 {
        self.close.min(self.open) - self.low
    }

    /// Is bullish candle (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Is bearish candle (close < open)
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Is doji (small body relative to range)
    pub fn is_doji(&self, threshold: f64) -> bool {
        if self.range() == 0.0 {
            return true;
        }
        self.body_size() / self.range() < threshold
    }

    /// Calculate typical price (HLC/3)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate VWAP contribution
    pub fn vwap_contribution(&self) -> f64 {
        self.typical_price() * self.volume
    }

    /// Get midpoint price
    pub fn midpoint(&self) -> f64 {
        (self.high + self.low) / 2.0
    }

    /// Get percentage change
    pub fn pct_change(&self) -> f64 {
        if self.open == 0.0 {
            0.0
        } else {
            (self.close - self.open) / self.open * 100.0
        }
    }

    /// Get log return
    pub fn log_return(&self) -> f64 {
        if self.open <= 0.0 || self.close <= 0.0 {
            0.0
        } else {
            (self.close / self.open).ln()
        }
    }
}

/// Builder for creating candles
#[derive(Debug, Default)]
pub struct CandleBuilder {
    symbol: Option<String>,
    timestamp: Option<DateTime<Utc>>,
    timeframe: Option<Timeframe>,
    open: Option<f64>,
    high: Option<f64>,
    low: Option<f64>,
    close: Option<f64>,
    volume: Option<f64>,
}

impl CandleBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set symbol
    pub fn symbol(mut self, symbol: &str) -> Self {
        self.symbol = Some(symbol.to_string());
        self
    }

    /// Set timestamp
    pub fn timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Set timeframe
    pub fn timeframe(mut self, timeframe: Timeframe) -> Self {
        self.timeframe = Some(timeframe);
        self
    }

    /// Set OHLC prices
    pub fn ohlc(mut self, open: f64, high: f64, low: f64, close: f64) -> Self {
        self.open = Some(open);
        self.high = Some(high);
        self.low = Some(low);
        self.close = Some(close);
        self
    }

    /// Set volume
    pub fn volume(mut self, volume: f64) -> Self {
        self.volume = Some(volume);
        self
    }

    /// Build the candle
    pub fn build(self) -> Option<Candle> {
        Some(Candle {
            symbol: self.symbol?,
            timestamp: self.timestamp?,
            timeframe: self.timeframe?,
            open: self.open?,
            high: self.high?,
            low: self.low?,
            close: self.close?,
            volume: self.volume.unwrap_or(0.0),
        })
    }
}

/// Collection of candles as a time series
#[derive(Debug, Clone)]
pub struct CandleSeries {
    /// Symbol
    pub symbol: String,
    /// Timeframe
    pub timeframe: Timeframe,
    /// Candles (sorted by timestamp)
    pub candles: Vec<Candle>,
}

impl CandleSeries {
    /// Create from vector of candles
    pub fn new(mut candles: Vec<Candle>) -> Option<Self> {
        if candles.is_empty() {
            return None;
        }

        candles.sort_by_key(|c| c.timestamp);

        let symbol = candles[0].symbol.clone();
        let timeframe = candles[0].timeframe;

        Some(Self {
            symbol,
            timeframe,
            candles,
        })
    }

    /// Get number of candles
    pub fn len(&self) -> usize {
        self.candles.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.candles.is_empty()
    }

    /// Get close prices as array
    pub fn close_prices(&self) -> Array1<f64> {
        Array1::from_iter(self.candles.iter().map(|c| c.close))
    }

    /// Get open prices as array
    pub fn open_prices(&self) -> Array1<f64> {
        Array1::from_iter(self.candles.iter().map(|c| c.open))
    }

    /// Get high prices as array
    pub fn high_prices(&self) -> Array1<f64> {
        Array1::from_iter(self.candles.iter().map(|c| c.high))
    }

    /// Get low prices as array
    pub fn low_prices(&self) -> Array1<f64> {
        Array1::from_iter(self.candles.iter().map(|c| c.low))
    }

    /// Get volumes as array
    pub fn volumes(&self) -> Array1<f64> {
        Array1::from_iter(self.candles.iter().map(|c| c.volume))
    }

    /// Get returns
    pub fn returns(&self) -> Array1<f64> {
        let close = self.close_prices();
        let mut returns = Array1::zeros(close.len());

        for i in 1..close.len() {
            if close[i - 1] != 0.0 {
                returns[i] = (close[i] - close[i - 1]) / close[i - 1];
            }
        }

        returns
    }

    /// Get log returns
    pub fn log_returns(&self) -> Array1<f64> {
        let close = self.close_prices();
        let mut returns = Array1::zeros(close.len());

        for i in 1..close.len() {
            if close[i - 1] > 0.0 && close[i] > 0.0 {
                returns[i] = (close[i] / close[i - 1]).ln();
            }
        }

        returns
    }

    /// Convert to feature matrix for model input
    ///
    /// Features: [open, high, low, close, volume, returns, log_returns, range, body]
    pub fn to_feature_matrix(&self) -> Array2<f64> {
        let n = self.len();
        let mut features = Array2::zeros((9, n));

        for (i, candle) in self.candles.iter().enumerate() {
            features[[0, i]] = candle.open;
            features[[1, i]] = candle.high;
            features[[2, i]] = candle.low;
            features[[3, i]] = candle.close;
            features[[4, i]] = candle.volume;

            // Calculate returns
            if i > 0 && self.candles[i - 1].close != 0.0 {
                features[[5, i]] = (candle.close - self.candles[i - 1].close)
                    / self.candles[i - 1].close;
                if candle.close > 0.0 && self.candles[i - 1].close > 0.0 {
                    features[[6, i]] = (candle.close / self.candles[i - 1].close).ln();
                }
            }

            features[[7, i]] = candle.range();
            features[[8, i]] = candle.body();
        }

        features
    }

    /// Normalize features for model input
    pub fn to_normalized_features(&self) -> Array2<f64> {
        let mut features = self.to_feature_matrix();

        // Normalize each feature row
        for mut row in features.rows_mut() {
            let mean = row.mean().unwrap_or(0.0);
            let std = row.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(1.0).sqrt();

            if std > 1e-10 {
                row.mapv_inplace(|x| (x - mean) / std);
            } else {
                row.mapv_inplace(|_| 0.0);
            }
        }

        features
    }

    /// Create sliding window samples
    pub fn create_windows(&self, window_size: usize) -> Vec<Array2<f64>> {
        let features = self.to_normalized_features();
        let n = features.dim().1;

        if n < window_size {
            return vec![];
        }

        (0..=n - window_size)
            .map(|i| features.slice(ndarray::s![.., i..i + window_size]).to_owned())
            .collect()
    }

    /// Create labeled samples for classification
    ///
    /// Label: 0 = down, 1 = neutral, 2 = up
    pub fn create_labeled_samples(
        &self,
        window_size: usize,
        threshold: f64,
    ) -> (Vec<Array2<f64>>, Vec<u8>) {
        let features = self.to_normalized_features();
        let n = features.dim().1;

        if n <= window_size {
            return (vec![], vec![]);
        }

        let mut samples = Vec::new();
        let mut labels = Vec::new();

        for i in 0..n - window_size {
            let window = features.slice(ndarray::s![.., i..i + window_size]).to_owned();

            // Calculate future return
            let current_close = self.candles[i + window_size - 1].close;
            let future_close = self.candles[i + window_size].close;

            if current_close > 0.0 {
                let return_pct = (future_close - current_close) / current_close * 100.0;

                let label = if return_pct > threshold {
                    2 // Up
                } else if return_pct < -threshold {
                    0 // Down
                } else {
                    1 // Neutral
                };

                samples.push(window);
                labels.push(label);
            }
        }

        (samples, labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candle() -> Candle {
        Candle::new(
            "BTCUSDT",
            Utc::now(),
            Timeframe::H1,
            100.0,
            110.0,
            95.0,
            105.0,
            1000.0,
        )
    }

    #[test]
    fn test_candle_calculations() {
        let candle = create_test_candle();

        assert_eq!(candle.body(), 5.0);
        assert_eq!(candle.body_size(), 5.0);
        assert_eq!(candle.range(), 15.0);
        assert_eq!(candle.upper_wick(), 5.0); // 110 - 105
        assert_eq!(candle.lower_wick(), 5.0); // 100 - 95
        assert!(candle.is_bullish());
        assert!(!candle.is_bearish());
    }

    #[test]
    fn test_timeframe_conversion() {
        assert_eq!(Timeframe::from_bybit_interval("60"), Some(Timeframe::H1));
        assert_eq!(Timeframe::H1.to_bybit_interval(), "60");
        assert_eq!(Timeframe::D1.as_seconds(), 86400);
    }

    #[test]
    fn test_candle_builder() {
        let candle = CandleBuilder::new()
            .symbol("ETHUSDT")
            .timestamp(Utc::now())
            .timeframe(Timeframe::M5)
            .ohlc(100.0, 105.0, 98.0, 103.0)
            .volume(500.0)
            .build();

        assert!(candle.is_some());
        let candle = candle.unwrap();
        assert_eq!(candle.symbol, "ETHUSDT");
        assert_eq!(candle.volume, 500.0);
    }

    #[test]
    fn test_candle_series() {
        let candles: Vec<Candle> = (0..10)
            .map(|i| {
                Candle::new(
                    "BTCUSDT",
                    Utc::now() + Duration::hours(i),
                    Timeframe::H1,
                    100.0 + i as f64,
                    105.0 + i as f64,
                    95.0 + i as f64,
                    102.0 + i as f64,
                    1000.0,
                )
            })
            .collect();

        let series = CandleSeries::new(candles).unwrap();
        assert_eq!(series.len(), 10);

        let close = series.close_prices();
        assert_eq!(close.len(), 10);

        let features = series.to_feature_matrix();
        assert_eq!(features.dim(), (9, 10));
    }
}
