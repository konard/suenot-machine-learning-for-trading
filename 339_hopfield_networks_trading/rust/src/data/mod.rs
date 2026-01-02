//! Data module for market data handling
//!
//! This module provides data structures and clients for fetching
//! cryptocurrency market data from the Bybit exchange.

pub mod bybit;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// OHLCV Candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Opening time
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
    /// Turnover (quote volume)
    pub turnover: f64,
}

impl Candle {
    /// Create a new Candle
    pub fn new(
        timestamp: DateTime<Utc>,
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

    /// Calculate the body size (absolute difference between open and close)
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Calculate the range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Check if the candle is bullish (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if the candle is bearish (close < open)
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Calculate upper shadow size
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.open.max(self.close)
    }

    /// Calculate lower shadow size
    pub fn lower_shadow(&self) -> f64 {
        self.open.min(self.close) - self.low
    }

    /// Calculate typical price (H+L+C)/3
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate VWAP approximation
    pub fn vwap(&self) -> f64 {
        self.typical_price()
    }

    /// Convert to feature vector for pattern encoding
    pub fn to_features(&self) -> Vec<f64> {
        vec![
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            self.body_size(),
            self.range(),
            self.upper_shadow(),
            self.lower_shadow(),
        ]
    }

    /// Normalize candle relative to its range
    pub fn normalize(&self) -> NormalizedCandle {
        let range = self.range().max(1e-10);
        NormalizedCandle {
            open: (self.open - self.low) / range,
            high: 1.0, // Always 1 after normalization
            low: 0.0,  // Always 0 after normalization
            close: (self.close - self.low) / range,
            volume_ratio: 1.0, // Needs context for proper normalization
        }
    }
}

/// Normalized candle with values in [0, 1]
#[derive(Debug, Clone)]
pub struct NormalizedCandle {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume_ratio: f64,
}

impl NormalizedCandle {
    /// Convert to feature vector
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume_ratio,
        ]
    }
}

/// Trade data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade ID
    pub id: String,
    /// Symbol
    pub symbol: String,
    /// Trade price
    pub price: f64,
    /// Trade quantity
    pub qty: f64,
    /// Trade side (Buy or Sell)
    pub side: TradeSide,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Trade side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Order book level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Price level
    pub price: f64,
    /// Quantity at this level
    pub qty: f64,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Symbol
    pub symbol: String,
    /// Bid levels (sorted by price descending)
    pub bids: Vec<OrderBookLevel>,
    /// Ask levels (sorted by price ascending)
    pub asks: Vec<OrderBookLevel>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
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

    /// Calculate the mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) => Some((ask + bid) / 2.0),
            _ => None,
        }
    }

    /// Calculate bid-ask imbalance
    pub fn imbalance(&self) -> f64 {
        let bid_volume: f64 = self.bids.iter().map(|l| l.qty).sum();
        let ask_volume: f64 = self.asks.iter().map(|l| l.qty).sum();
        let total = bid_volume + ask_volume;

        if total < 1e-10 {
            return 0.0;
        }

        (bid_volume - ask_volume) / total
    }
}

/// Time intervals for candle data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interval {
    Min1,
    Min3,
    Min5,
    Min15,
    Min30,
    Hour1,
    Hour2,
    Hour4,
    Hour6,
    Hour12,
    Day1,
    Week1,
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

    /// Get interval duration in minutes
    pub fn minutes(&self) -> u64 {
        match self {
            Interval::Min1 => 1,
            Interval::Min3 => 3,
            Interval::Min5 => 5,
            Interval::Min15 => 15,
            Interval::Min30 => 30,
            Interval::Hour1 => 60,
            Interval::Hour2 => 120,
            Interval::Hour4 => 240,
            Interval::Hour6 => 360,
            Interval::Hour12 => 720,
            Interval::Day1 => 1440,
            Interval::Week1 => 10080,
            Interval::Month1 => 43200,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_properties() {
        let candle = Candle::new(
            Utc::now(),
            100.0, // open
            110.0, // high
            95.0,  // low
            105.0, // close
            1000.0,
            100000.0,
        );

        assert!(candle.is_bullish());
        assert!(!candle.is_bearish());
        assert_eq!(candle.body_size(), 5.0);
        assert_eq!(candle.range(), 15.0);
        assert_eq!(candle.upper_shadow(), 5.0);
        assert_eq!(candle.lower_shadow(), 5.0);
    }

    #[test]
    fn test_order_book() {
        let orderbook = OrderBook {
            symbol: "BTCUSDT".to_string(),
            bids: vec![
                OrderBookLevel { price: 100.0, qty: 10.0 },
                OrderBookLevel { price: 99.0, qty: 20.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 101.0, qty: 15.0 },
                OrderBookLevel { price: 102.0, qty: 25.0 },
            ],
            timestamp: Utc::now(),
        };

        assert_eq!(orderbook.best_bid(), Some(100.0));
        assert_eq!(orderbook.best_ask(), Some(101.0));
        assert_eq!(orderbook.spread(), Some(1.0));
        assert_eq!(orderbook.mid_price(), Some(100.5));
    }
}
