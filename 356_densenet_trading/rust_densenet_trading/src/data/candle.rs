//! Candle (OHLCV) Data Structure
//!
//! Represents a single candlestick with open, high, low, close prices and volume.

use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};

/// A single candlestick (OHLCV data)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Unix timestamp in milliseconds
    pub timestamp: u64,

    /// Opening price
    pub open: f64,

    /// Highest price
    pub high: f64,

    /// Lowest price
    pub low: f64,

    /// Closing price
    pub close: f64,

    /// Trading volume (in base currency)
    pub volume: f64,

    /// Turnover (in quote currency)
    pub turnover: f64,
}

impl Candle {
    /// Create a new candle
    pub fn new(
        timestamp: u64,
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

    /// Get the datetime representation
    pub fn datetime(&self) -> DateTime<Utc> {
        Utc.timestamp_millis_opt(self.timestamp as i64)
            .single()
            .unwrap_or_else(|| Utc::now())
    }

    /// Check if this is a bullish candle (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if this is a bearish candle (close < open)
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Get the body size (absolute difference between open and close)
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Get the body as percentage of price
    pub fn body_percent(&self) -> f64 {
        self.body_size() / self.open * 100.0
    }

    /// Get the range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Get the upper shadow (wick)
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.close.max(self.open)
    }

    /// Get the lower shadow (wick)
    pub fn lower_shadow(&self) -> f64 {
        self.close.min(self.open) - self.low
    }

    /// Get the typical price (average of high, low, close)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Get the VWAP-like price (weighted average)
    pub fn weighted_close(&self) -> f64 {
        (self.high + self.low + 2.0 * self.close) / 4.0
    }

    /// Check if this is a doji (small body relative to range)
    pub fn is_doji(&self, threshold: f64) -> bool {
        let body_ratio = self.body_size() / self.range();
        body_ratio < threshold
    }

    /// Check if this is a hammer pattern
    pub fn is_hammer(&self) -> bool {
        let body = self.body_size();
        let lower = self.lower_shadow();
        let upper = self.upper_shadow();

        lower >= 2.0 * body && upper <= body * 0.3
    }

    /// Check if this is an inverted hammer
    pub fn is_inverted_hammer(&self) -> bool {
        let body = self.body_size();
        let lower = self.lower_shadow();
        let upper = self.upper_shadow();

        upper >= 2.0 * body && lower <= body * 0.3
    }

    /// Calculate returns from previous candle
    pub fn returns_from(&self, previous: &Candle) -> f64 {
        (self.close / previous.close).ln()
    }
}

/// A collection of candles
#[derive(Debug, Clone, Default)]
pub struct CandleSeries {
    pub candles: Vec<Candle>,
}

impl CandleSeries {
    /// Create a new empty series
    pub fn new() -> Self {
        Self {
            candles: Vec::new(),
        }
    }

    /// Create from a vector of candles
    pub fn from_vec(candles: Vec<Candle>) -> Self {
        Self { candles }
    }

    /// Add a candle
    pub fn push(&mut self, candle: Candle) {
        self.candles.push(candle);
    }

    /// Get the number of candles
    pub fn len(&self) -> usize {
        self.candles.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.candles.is_empty()
    }

    /// Get closing prices
    pub fn closes(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.close).collect()
    }

    /// Get opening prices
    pub fn opens(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.open).collect()
    }

    /// Get high prices
    pub fn highs(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.high).collect()
    }

    /// Get low prices
    pub fn lows(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.low).collect()
    }

    /// Get volumes
    pub fn volumes(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.volume).collect()
    }

    /// Get returns series
    pub fn returns(&self) -> Vec<f64> {
        if self.candles.len() < 2 {
            return Vec::new();
        }

        self.candles
            .windows(2)
            .map(|w| w[1].returns_from(&w[0]))
            .collect()
    }

    /// Get the last n candles
    pub fn tail(&self, n: usize) -> Vec<&Candle> {
        let start = if n >= self.candles.len() {
            0
        } else {
            self.candles.len() - n
        };
        self.candles[start..].iter().collect()
    }

    /// Sort by timestamp
    pub fn sort_by_time(&mut self) {
        self.candles.sort_by_key(|c| c.timestamp);
    }

    /// Resample to a higher timeframe
    pub fn resample(&self, factor: usize) -> CandleSeries {
        if factor <= 1 || self.candles.is_empty() {
            return self.clone();
        }

        let mut resampled = Vec::new();

        for chunk in self.candles.chunks(factor) {
            if chunk.is_empty() {
                continue;
            }

            let open = chunk[0].open;
            let close = chunk.last().unwrap().close;
            let high = chunk.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max);
            let low = chunk.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
            let volume: f64 = chunk.iter().map(|c| c.volume).sum();
            let turnover: f64 = chunk.iter().map(|c| c.turnover).sum();
            let timestamp = chunk[0].timestamp;

            resampled.push(Candle {
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                turnover,
            });
        }

        CandleSeries::from_vec(resampled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_candle() -> Candle {
        Candle::new(
            1704067200000, // 2024-01-01
            42000.0,
            42500.0,
            41800.0,
            42300.0,
            1000.0,
            42000000.0,
        )
    }

    #[test]
    fn test_candle_creation() {
        let candle = sample_candle();
        assert_eq!(candle.open, 42000.0);
        assert_eq!(candle.close, 42300.0);
    }

    #[test]
    fn test_bullish_bearish() {
        let bullish = Candle::new(0, 100.0, 110.0, 95.0, 105.0, 100.0, 10000.0);
        assert!(bullish.is_bullish());
        assert!(!bullish.is_bearish());

        let bearish = Candle::new(0, 105.0, 110.0, 95.0, 100.0, 100.0, 10000.0);
        assert!(bearish.is_bearish());
        assert!(!bearish.is_bullish());
    }

    #[test]
    fn test_candle_metrics() {
        let candle = sample_candle();

        assert!((candle.body_size() - 300.0).abs() < 0.01);
        assert!((candle.range() - 700.0).abs() < 0.01);
        assert!((candle.upper_shadow() - 200.0).abs() < 0.01);
        assert!((candle.lower_shadow() - 200.0).abs() < 0.01);
    }

    #[test]
    fn test_candle_series() {
        let mut series = CandleSeries::new();
        series.push(Candle::new(1, 100.0, 105.0, 98.0, 102.0, 100.0, 10000.0));
        series.push(Candle::new(2, 102.0, 108.0, 101.0, 106.0, 150.0, 15000.0));

        assert_eq!(series.len(), 2);
        assert_eq!(series.closes(), vec![102.0, 106.0]);

        let returns = series.returns();
        assert_eq!(returns.len(), 1);
    }

    #[test]
    fn test_resample() {
        let candles = vec![
            Candle::new(1, 100.0, 105.0, 98.0, 102.0, 100.0, 10000.0),
            Candle::new(2, 102.0, 108.0, 101.0, 106.0, 150.0, 15000.0),
            Candle::new(3, 106.0, 110.0, 104.0, 108.0, 120.0, 12000.0),
            Candle::new(4, 108.0, 112.0, 106.0, 110.0, 130.0, 13000.0),
        ];

        let series = CandleSeries::from_vec(candles);
        let resampled = series.resample(2);

        assert_eq!(resampled.len(), 2);
        assert_eq!(resampled.candles[0].open, 100.0);
        assert_eq!(resampled.candles[0].close, 106.0);
        assert_eq!(resampled.candles[0].high, 108.0);
        assert_eq!(resampled.candles[0].volume, 250.0);
    }
}
