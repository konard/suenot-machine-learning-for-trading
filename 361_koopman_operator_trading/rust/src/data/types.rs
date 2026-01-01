//! Core data types for market data

use serde::{Deserialize, Serialize};

/// OHLCV Candle data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Timestamp in milliseconds
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
    /// Turnover (quote volume)
    pub turnover: f64,
}

impl Candle {
    /// Calculate typical price (HLC/3)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate VWAP approximation
    pub fn vwap(&self) -> f64 {
        self.turnover / self.volume
    }

    /// Get the candle range
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Get the body size (absolute)
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Check if candle is bullish
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
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
    /// Trading symbol
    pub symbol: String,
    /// Timestamp in milliseconds
    pub timestamp: u64,
    /// Bid levels (sorted by price descending)
    pub bids: Vec<OrderBookLevel>,
    /// Ask levels (sorted by price ascending)
    pub asks: Vec<OrderBookLevel>,
}

impl OrderBook {
    /// Get best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Get mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Get spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get order book imbalance
    pub fn imbalance(&self, depth: usize) -> f64 {
        let bid_volume: f64 = self.bids.iter().take(depth).map(|l| l.quantity).sum();
        let ask_volume: f64 = self.asks.iter().take(depth).map(|l| l.quantity).sum();
        let total = bid_volume + ask_volume;
        if total > 0.0 {
            (bid_volume - ask_volume) / total
        } else {
            0.0
        }
    }
}

/// Trade side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Individual trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade ID
    pub id: String,
    /// Trading symbol
    pub symbol: String,
    /// Trade price
    pub price: f64,
    /// Trade quantity
    pub quantity: f64,
    /// Trade side
    pub side: TradeSide,
    /// Timestamp in milliseconds
    pub timestamp: u64,
}

/// Time series data container
#[derive(Debug, Clone)]
pub struct TimeSeries {
    /// Values
    pub values: Vec<f64>,
    /// Timestamps (optional)
    pub timestamps: Option<Vec<u64>>,
    /// Name/label
    pub name: String,
}

impl TimeSeries {
    /// Create new time series from values
    pub fn new(values: Vec<f64>, name: &str) -> Self {
        Self {
            values,
            timestamps: None,
            name: name.to_string(),
        }
    }

    /// Create time series with timestamps
    pub fn with_timestamps(values: Vec<f64>, timestamps: Vec<u64>, name: &str) -> Self {
        Self {
            values,
            timestamps: Some(timestamps),
            name: name.to_string(),
        }
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get returns
    pub fn returns(&self) -> Vec<f64> {
        self.values
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    /// Get log returns
    pub fn log_returns(&self) -> Vec<f64> {
        self.values
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect()
    }

    /// Calculate mean
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    /// Calculate standard deviation
    pub fn std(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance = self.values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (self.values.len() - 1) as f64;
        variance.sqrt()
    }

    /// Normalize to zero mean and unit variance
    pub fn normalize(&self) -> Vec<f64> {
        let mean = self.mean();
        let std = self.std();
        if std < 1e-10 {
            return vec![0.0; self.values.len()];
        }
        self.values.iter().map(|x| (x - mean) / std).collect()
    }
}
