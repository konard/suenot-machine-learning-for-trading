//! Core data types for market data
//!
//! This module defines the fundamental data structures used throughout the library:
//! - Candle: OHLCV candlestick data
//! - OrderBook: Market depth information
//! - Trade: Individual trade records
//! - Dataset: Features and labels for ML

use chrono::{DateTime, TimeZone, Utc};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// OHLCV candlestick data
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
    /// Trading volume (base currency)
    pub volume: f64,
    /// Turnover (quote currency)
    pub turnover: f64,
}

impl Candle {
    /// Get the candle's datetime
    pub fn datetime(&self) -> DateTime<Utc> {
        Utc.timestamp_millis_opt(self.timestamp as i64)
            .single()
            .unwrap_or_else(Utc::now)
    }

    /// Calculate the candle's body size (|close - open|)
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Calculate the candle's total range (high - low)
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

    /// Calculate the percentage change
    pub fn pct_change(&self) -> f64 {
        if self.open == 0.0 {
            0.0
        } else {
            (self.close - self.open) / self.open * 100.0
        }
    }

    /// Calculate the upper shadow size
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.close.max(self.open)
    }

    /// Calculate the lower shadow size
    pub fn lower_shadow(&self) -> f64 {
        self.close.min(self.open) - self.low
    }

    /// Get typical price (HLC/3)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Get VWAP-like value for single candle
    pub fn vwap(&self) -> f64 {
        if self.volume == 0.0 {
            self.typical_price()
        } else {
            self.turnover / self.volume
        }
    }
}

/// Order book level (price and quantity)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub timestamp: u64,
    pub bids: Vec<OrderBookLevel>,
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

    /// Calculate the bid-ask spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Calculate the bid-ask spread as percentage
    pub fn spread_pct(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) if bid > 0.0 => Some((ask - bid) / bid * 100.0),
            _ => None,
        }
    }

    /// Calculate mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) => Some((ask + bid) / 2.0),
            _ => None,
        }
    }

    /// Calculate total bid volume
    pub fn total_bid_volume(&self) -> f64 {
        self.bids.iter().map(|l| l.quantity).sum()
    }

    /// Calculate total ask volume
    pub fn total_ask_volume(&self) -> f64 {
        self.asks.iter().map(|l| l.quantity).sum()
    }

    /// Calculate order book imbalance (bid_vol - ask_vol) / (bid_vol + ask_vol)
    pub fn imbalance(&self) -> f64 {
        let bid_vol = self.total_bid_volume();
        let ask_vol = self.total_ask_volume();
        let total = bid_vol + ask_vol;
        if total == 0.0 {
            0.0
        } else {
            (bid_vol - ask_vol) / total
        }
    }
}

/// Trade side (buy or sell)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Individual trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: String,
    pub symbol: String,
    pub price: f64,
    pub quantity: f64,
    pub side: TradeSide,
    pub timestamp: u64,
}

impl Trade {
    /// Get the trade value (price * quantity)
    pub fn value(&self) -> f64 {
        self.price * self.quantity
    }

    /// Get the trade datetime
    pub fn datetime(&self) -> DateTime<Utc> {
        Utc.timestamp_millis_opt(self.timestamp as i64)
            .single()
            .unwrap_or_else(Utc::now)
    }
}

/// Dataset for machine learning
///
/// Contains feature matrix X and target vector y
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Feature matrix (n_samples x n_features)
    pub x: Array2<f64>,
    /// Target vector (n_samples)
    pub y: Array1<f64>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Target name
    pub target_name: String,
}

impl Dataset {
    /// Create a new dataset
    pub fn new(
        x: Array2<f64>,
        y: Array1<f64>,
        feature_names: Vec<String>,
        target_name: String,
    ) -> Self {
        assert_eq!(x.nrows(), y.len(), "X rows must match y length");
        Self {
            x,
            y,
            feature_names,
            target_name,
        }
    }

    /// Get number of samples
    pub fn n_samples(&self) -> usize {
        self.x.nrows()
    }

    /// Get number of features
    pub fn n_features(&self) -> usize {
        self.x.ncols()
    }

    /// Split dataset into train and test sets
    pub fn train_test_split(&self, test_ratio: f64) -> (Dataset, Dataset) {
        let n = self.n_samples();
        let test_size = (n as f64 * test_ratio).round() as usize;
        let train_size = n - test_size;

        let x_train = self.x.slice(ndarray::s![..train_size, ..]).to_owned();
        let y_train = self.y.slice(ndarray::s![..train_size]).to_owned();

        let x_test = self.x.slice(ndarray::s![train_size.., ..]).to_owned();
        let y_test = self.y.slice(ndarray::s![train_size..]).to_owned();

        let train = Dataset::new(
            x_train,
            y_train,
            self.feature_names.clone(),
            self.target_name.clone(),
        );

        let test = Dataset::new(
            x_test,
            y_test,
            self.feature_names.clone(),
            self.target_name.clone(),
        );

        (train, test)
    }

    /// Get a subset of features by indices
    pub fn select_features(&self, indices: &[usize]) -> Dataset {
        let x_new = self.x.select(ndarray::Axis(1), indices);
        let feature_names: Vec<String> = indices
            .iter()
            .map(|&i| self.feature_names[i].clone())
            .collect();

        Dataset::new(x_new, self.y.clone(), feature_names, self.target_name.clone())
    }

    /// Normalize features to [0, 1] range
    pub fn normalize(&mut self) {
        for mut col in self.x.columns_mut() {
            let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = max - min;
            if range > 0.0 {
                col.mapv_inplace(|x| (x - min) / range);
            }
        }
    }

    /// Standardize features (zero mean, unit variance)
    pub fn standardize(&mut self) {
        for mut col in self.x.columns_mut() {
            let mean = col.mean().unwrap_or(0.0);
            let std = col.std(0.0);
            if std > 0.0 {
                col.mapv_inplace(|x| (x - mean) / std);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_candle_is_bullish() {
        let candle = Candle {
            timestamp: 0,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };
        assert!(candle.is_bullish());
        assert!(!candle.is_bearish());
    }

    #[test]
    fn test_orderbook_spread() {
        let book = OrderBook {
            symbol: "BTCUSDT".to_string(),
            timestamp: 0,
            bids: vec![OrderBookLevel { price: 99.0, quantity: 10.0 }],
            asks: vec![OrderBookLevel { price: 101.0, quantity: 10.0 }],
        };
        assert_eq!(book.spread(), Some(2.0));
        assert_eq!(book.mid_price(), Some(100.0));
    }

    #[test]
    fn test_dataset_split() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0];
        let dataset = Dataset::new(
            x,
            y,
            vec!["f1".to_string(), "f2".to_string()],
            "target".to_string(),
        );

        let (train, test) = dataset.train_test_split(0.4);
        assert_eq!(train.n_samples(), 3);
        assert_eq!(test.n_samples(), 2);
    }
}
