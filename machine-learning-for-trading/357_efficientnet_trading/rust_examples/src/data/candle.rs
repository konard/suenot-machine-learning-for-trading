//! Candlestick/OHLCV data structures

use serde::{Deserialize, Serialize};

/// OHLCV candlestick data
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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
    /// Trading volume
    pub volume: f64,
}

impl Candle {
    /// Create a new candle
    pub fn new(timestamp: u64, open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Check if candle is bullish (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close >= self.open
    }

    /// Check if candle is bearish (close < open)
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Get the body size (absolute difference between open and close)
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Get the full range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Get upper wick size
    pub fn upper_wick(&self) -> f64 {
        self.high - self.open.max(self.close)
    }

    /// Get lower wick size
    pub fn lower_wick(&self) -> f64 {
        self.open.min(self.close) - self.low
    }

    /// Get midpoint price
    pub fn midpoint(&self) -> f64 {
        (self.high + self.low) / 2.0
    }

    /// Get typical price (HLC average)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Get price change
    pub fn change(&self) -> f64 {
        self.close - self.open
    }

    /// Get price change percentage
    pub fn change_percent(&self) -> f64 {
        if self.open == 0.0 {
            0.0
        } else {
            (self.close - self.open) / self.open * 100.0
        }
    }

    /// Check if this is a doji (small body)
    pub fn is_doji(&self, threshold: f64) -> bool {
        self.body_size() / self.range() < threshold
    }

    /// Check if this is a hammer pattern
    pub fn is_hammer(&self) -> bool {
        let body = self.body_size();
        let range = self.range();
        if range == 0.0 {
            return false;
        }
        let lower = self.lower_wick();
        let upper = self.upper_wick();

        lower >= 2.0 * body && upper <= body * 0.3
    }

    /// Check if this is an inverted hammer
    pub fn is_inverted_hammer(&self) -> bool {
        let body = self.body_size();
        let range = self.range();
        if range == 0.0 {
            return false;
        }
        let lower = self.lower_wick();
        let upper = self.upper_wick();

        upper >= 2.0 * body && lower <= body * 0.3
    }
}

/// Calculate statistics for a series of candles
pub struct CandleStats {
    pub count: usize,
    pub min_price: f64,
    pub max_price: f64,
    pub avg_volume: f64,
    pub total_volume: f64,
    pub volatility: f64,
}

impl CandleStats {
    /// Calculate statistics from candles
    pub fn from_candles(candles: &[Candle]) -> Self {
        if candles.is_empty() {
            return Self {
                count: 0,
                min_price: 0.0,
                max_price: 0.0,
                avg_volume: 0.0,
                total_volume: 0.0,
                volatility: 0.0,
            };
        }

        let count = candles.len();
        let min_price = candles.iter().map(|c| c.low).fold(f64::MAX, f64::min);
        let max_price = candles.iter().map(|c| c.high).fold(f64::MIN, f64::max);
        let total_volume: f64 = candles.iter().map(|c| c.volume).sum();
        let avg_volume = total_volume / count as f64;

        // Calculate volatility as standard deviation of returns
        let returns: Vec<f64> = candles
            .windows(2)
            .map(|w| (w[1].close / w[0].close).ln())
            .collect();

        let volatility = if returns.len() > 1 {
            let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 =
                returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        Self {
            count,
            min_price,
            max_price,
            avg_volume,
            total_volume,
            volatility,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_bullish() {
        let candle = Candle::new(0, 100.0, 110.0, 95.0, 105.0, 1000.0);
        assert!(candle.is_bullish());
        assert!(!candle.is_bearish());
    }

    #[test]
    fn test_candle_bearish() {
        let candle = Candle::new(0, 105.0, 110.0, 95.0, 100.0, 1000.0);
        assert!(!candle.is_bullish());
        assert!(candle.is_bearish());
    }

    #[test]
    fn test_candle_metrics() {
        let candle = Candle::new(0, 100.0, 120.0, 90.0, 110.0, 1000.0);
        assert_eq!(candle.body_size(), 10.0);
        assert_eq!(candle.range(), 30.0);
        assert_eq!(candle.upper_wick(), 10.0);
        assert_eq!(candle.lower_wick(), 10.0);
    }

    #[test]
    fn test_change_percent() {
        let candle = Candle::new(0, 100.0, 110.0, 95.0, 110.0, 1000.0);
        assert_eq!(candle.change_percent(), 10.0);
    }
}
