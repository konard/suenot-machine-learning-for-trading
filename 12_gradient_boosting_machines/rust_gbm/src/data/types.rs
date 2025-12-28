//! Data types for cryptocurrency market data
//!
//! This module defines the core data structures used throughout the project.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// OHLCV (Open, High, Low, Close, Volume) candle data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Candle open time
    pub timestamp: DateTime<Utc>,
    /// Symbol (e.g., "BTCUSDT")
    pub symbol: String,
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

impl Candle {
    /// Calculate the candle body size (absolute difference between open and close)
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Calculate the upper shadow size
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.open.max(self.close)
    }

    /// Calculate the lower shadow size
    pub fn lower_shadow(&self) -> f64 {
        self.open.min(self.close) - self.low
    }

    /// Check if the candle is bullish (green)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Calculate the return from open to close
    pub fn return_pct(&self) -> f64 {
        if self.open != 0.0 {
            (self.close - self.open) / self.open * 100.0
        } else {
            0.0
        }
    }
}

/// Time interval for candles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Interval {
    /// 1 minute
    Min1,
    /// 3 minutes
    Min3,
    /// 5 minutes
    Min5,
    /// 15 minutes
    Min15,
    /// 30 minutes
    Min30,
    /// 1 hour
    Hour1,
    /// 2 hours
    Hour2,
    /// 4 hours
    Hour4,
    /// 6 hours
    Hour6,
    /// 12 hours
    Hour12,
    /// 1 day
    Day1,
    /// 1 week
    Week1,
    /// 1 month
    Month1,
}

impl Interval {
    /// Convert to Bybit API interval string
    pub fn to_bybit_string(&self) -> &'static str {
        match self {
            Interval::Min1 => "1",
            Interval::Min3 => "3",
            Interval::Min5 => "5",
            Interval::Min15 => "15",
            Interval::Min30 => "30",
            Interval::Hour1 => "60",
            Interval::Hour2 => "120",
            Interval::Hour4 => "240",
            Interval::Hour6 => "360",
            Interval::Hour12 => "720",
            Interval::Day1 => "D",
            Interval::Week1 => "W",
            Interval::Month1 => "M",
        }
    }

    /// Get interval duration in milliseconds
    pub fn duration_ms(&self) -> i64 {
        match self {
            Interval::Min1 => 60 * 1000,
            Interval::Min3 => 3 * 60 * 1000,
            Interval::Min5 => 5 * 60 * 1000,
            Interval::Min15 => 15 * 60 * 1000,
            Interval::Min30 => 30 * 60 * 1000,
            Interval::Hour1 => 60 * 60 * 1000,
            Interval::Hour2 => 2 * 60 * 60 * 1000,
            Interval::Hour4 => 4 * 60 * 60 * 1000,
            Interval::Hour6 => 6 * 60 * 60 * 1000,
            Interval::Hour12 => 12 * 60 * 60 * 1000,
            Interval::Day1 => 24 * 60 * 60 * 1000,
            Interval::Week1 => 7 * 24 * 60 * 60 * 1000,
            Interval::Month1 => 30 * 24 * 60 * 60 * 1000,
        }
    }
}

/// Order book level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Price level
    pub price: f64,
    /// Quantity at this price
    pub quantity: f64,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Symbol
    pub symbol: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Bid levels (buy orders)
    pub bids: Vec<OrderBookLevel>,
    /// Ask levels (sell orders)
    pub asks: Vec<OrderBookLevel>,
}

impl OrderBook {
    /// Get the best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Get the best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Calculate the spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Calculate the spread percentage
    pub fn spread_pct(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) if bid > 0.0 => Some((ask - bid) / bid * 100.0),
            _ => None,
        }
    }

    /// Calculate the mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) => Some((ask + bid) / 2.0),
            _ => None,
        }
    }
}

/// Trade execution data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade ID
    pub id: String,
    /// Symbol
    pub symbol: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Price
    pub price: f64,
    /// Quantity
    pub quantity: f64,
    /// Trade side (buy or sell)
    pub side: TradeSide,
}

/// Trade side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Dataset for machine learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    /// Feature names
    pub feature_names: Vec<String>,
    /// Feature matrix (rows = samples, cols = features)
    pub features: Vec<Vec<f64>>,
    /// Target values
    pub targets: Vec<f64>,
    /// Timestamps for each sample
    pub timestamps: Vec<DateTime<Utc>>,
    /// Symbol
    pub symbol: String,
}

impl Dataset {
    /// Create a new empty dataset
    pub fn new(symbol: String, feature_names: Vec<String>) -> Self {
        Self {
            feature_names,
            features: Vec::new(),
            targets: Vec::new(),
            timestamps: Vec::new(),
            symbol,
        }
    }

    /// Add a sample to the dataset
    pub fn add_sample(&mut self, features: Vec<f64>, target: f64, timestamp: DateTime<Utc>) {
        self.features.push(features);
        self.targets.push(target);
        self.timestamps.push(timestamp);
    }

    /// Get the number of samples
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Check if the dataset is empty
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Get the number of features
    pub fn num_features(&self) -> usize {
        self.feature_names.len()
    }

    /// Split the dataset into training and test sets
    pub fn train_test_split(&self, train_ratio: f64) -> (Dataset, Dataset) {
        let split_idx = (self.len() as f64 * train_ratio) as usize;

        let train = Dataset {
            feature_names: self.feature_names.clone(),
            features: self.features[..split_idx].to_vec(),
            targets: self.targets[..split_idx].to_vec(),
            timestamps: self.timestamps[..split_idx].to_vec(),
            symbol: self.symbol.clone(),
        };

        let test = Dataset {
            feature_names: self.feature_names.clone(),
            features: self.features[split_idx..].to_vec(),
            targets: self.targets[split_idx..].to_vec(),
            timestamps: self.timestamps[split_idx..].to_vec(),
            symbol: self.symbol.clone(),
        };

        (train, test)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_calculations() {
        let candle = Candle {
            timestamp: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        assert!(candle.is_bullish());
        assert_eq!(candle.body_size(), 5.0);
        assert_eq!(candle.upper_shadow(), 5.0);
        assert_eq!(candle.lower_shadow(), 5.0);
        assert_eq!(candle.return_pct(), 5.0);
    }

    #[test]
    fn test_interval_conversion() {
        assert_eq!(Interval::Min1.to_bybit_string(), "1");
        assert_eq!(Interval::Hour1.to_bybit_string(), "60");
        assert_eq!(Interval::Day1.to_bybit_string(), "D");
    }
}
