//! # Market Snapshot
//!
//! Combined market state at a point in time for feature calculation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::orderbook::OrderBook;
use super::trade::TradeStats;

/// Complete market snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSnapshot {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Symbol
    pub symbol: String,
    /// Order book state
    pub orderbook: OrderBookSnapshot,
    /// Trade statistics for the period
    pub trade_stats: TradeStatsSnapshot,
    /// Calculated OFI value
    pub ofi: f64,
    /// Cumulative OFI
    pub cumulative_ofi: f64,
    /// VPIN value
    pub vpin: Option<f64>,
}

/// Simplified order book snapshot for features
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    /// Best bid price
    pub best_bid: f64,
    /// Best ask price
    pub best_ask: f64,
    /// Best bid size
    pub bid_size_l1: f64,
    /// Best ask size
    pub ask_size_l1: f64,
    /// Bid depth (5 levels)
    pub bid_depth_l5: f64,
    /// Ask depth (5 levels)
    pub ask_depth_l5: f64,
    /// Bid depth (10 levels)
    pub bid_depth_l10: f64,
    /// Ask depth (10 levels)
    pub ask_depth_l10: f64,
    /// Spread in price
    pub spread: f64,
    /// Spread in bps
    pub spread_bps: f64,
    /// Mid price
    pub mid_price: f64,
    /// Depth imbalance L1
    pub depth_imbalance_l1: f64,
    /// Depth imbalance L5
    pub depth_imbalance_l5: f64,
    /// Depth imbalance L10
    pub depth_imbalance_l10: f64,
    /// Weighted depth imbalance
    pub weighted_imbalance: f64,
    /// Bid slope
    pub bid_slope: f64,
    /// Ask slope
    pub ask_slope: f64,
}

impl OrderBookSnapshot {
    /// Create from an OrderBook
    pub fn from_orderbook(ob: &OrderBook) -> Self {
        let best_bid = ob.best_bid().map(|l| l.price).unwrap_or(0.0);
        let best_ask = ob.best_ask().map(|l| l.price).unwrap_or(0.0);
        let mid_price = ob.mid_price().unwrap_or(0.0);
        let spread = ob.spread().unwrap_or(0.0);
        let spread_bps = ob.spread_bps().unwrap_or(0.0);

        Self {
            best_bid,
            best_ask,
            bid_size_l1: ob.best_bid().map(|l| l.size).unwrap_or(0.0),
            ask_size_l1: ob.best_ask().map(|l| l.size).unwrap_or(0.0),
            bid_depth_l5: ob.bid_depth(5),
            ask_depth_l5: ob.ask_depth(5),
            bid_depth_l10: ob.bid_depth(10),
            ask_depth_l10: ob.ask_depth(10),
            spread,
            spread_bps,
            mid_price,
            depth_imbalance_l1: ob.depth_imbalance(1),
            depth_imbalance_l5: ob.depth_imbalance(5),
            depth_imbalance_l10: ob.depth_imbalance(10),
            weighted_imbalance: ob.weighted_depth_imbalance(10).unwrap_or(0.0),
            bid_slope: ob.bid_slope(5).unwrap_or(0.0),
            ask_slope: ob.ask_slope(5).unwrap_or(0.0),
        }
    }

    /// Calculate slope asymmetry
    pub fn slope_asymmetry(&self) -> f64 {
        let total = self.bid_slope.abs() + self.ask_slope.abs();
        if total > 0.0 {
            (self.bid_slope.abs() - self.ask_slope.abs()) / total
        } else {
            0.0
        }
    }
}

/// Simplified trade stats snapshot for features
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TradeStatsSnapshot {
    /// Trade count
    pub trade_count: usize,
    /// Total volume
    pub volume: f64,
    /// Buy volume
    pub buy_volume: f64,
    /// Sell volume
    pub sell_volume: f64,
    /// Trade imbalance
    pub trade_imbalance: f64,
    /// Count imbalance
    pub count_imbalance: f64,
    /// VWAP
    pub vwap: f64,
    /// Average trade size
    pub avg_trade_size: f64,
    /// Max trade size
    pub max_trade_size: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Return percentage
    pub return_pct: f64,
}

impl TradeStatsSnapshot {
    /// Create from TradeStats
    pub fn from_stats(stats: &TradeStats) -> Self {
        Self {
            trade_count: stats.count,
            volume: stats.volume,
            buy_volume: stats.buy_volume,
            sell_volume: stats.sell_volume,
            trade_imbalance: stats.trade_imbalance(),
            count_imbalance: stats.count_imbalance(),
            vwap: stats.vwap,
            avg_trade_size: stats.avg_size,
            max_trade_size: stats.max_size,
            high: stats.high,
            low: stats.low,
            return_pct: stats.return_pct(),
        }
    }

    /// Check if there's a large trade (> 2x average)
    pub fn has_large_trade(&self) -> bool {
        if self.avg_trade_size > 0.0 {
            self.max_trade_size > 2.0 * self.avg_trade_size
        } else {
            false
        }
    }

    /// Get buy aggression ratio
    pub fn buy_aggression(&self) -> f64 {
        if self.trade_count > 0 {
            self.buy_volume / self.volume
        } else {
            0.5
        }
    }
}

/// Time window for aggregation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeWindow {
    Seconds(u32),
    Minutes(u32),
    Hours(u32),
}

impl TimeWindow {
    /// Get duration in seconds
    pub fn as_seconds(&self) -> u32 {
        match self {
            TimeWindow::Seconds(s) => *s,
            TimeWindow::Minutes(m) => m * 60,
            TimeWindow::Hours(h) => h * 3600,
        }
    }

    /// Get duration in milliseconds
    pub fn as_millis(&self) -> u64 {
        self.as_seconds() as u64 * 1000
    }
}

/// Rolling window buffer for time series data
#[derive(Debug, Clone)]
pub struct RollingBuffer<T> {
    data: Vec<(DateTime<Utc>, T)>,
    window: TimeWindow,
    max_size: usize,
}

impl<T: Clone> RollingBuffer<T> {
    /// Create a new rolling buffer
    pub fn new(window: TimeWindow, max_size: usize) -> Self {
        Self {
            data: Vec::with_capacity(max_size),
            window,
            max_size,
        }
    }

    /// Add a new item
    pub fn push(&mut self, timestamp: DateTime<Utc>, item: T) {
        self.data.push((timestamp, item));

        // Remove old items
        self.cleanup(timestamp);

        // Limit size
        if self.data.len() > self.max_size {
            self.data.remove(0);
        }
    }

    /// Get all items in the current window
    pub fn get_window(&self, current_time: DateTime<Utc>) -> Vec<&T> {
        let window_start = current_time
            - chrono::Duration::seconds(self.window.as_seconds() as i64);

        self.data
            .iter()
            .filter(|(ts, _)| *ts >= window_start)
            .map(|(_, item)| item)
            .collect()
    }

    /// Get count of items in window
    pub fn count(&self, current_time: DateTime<Utc>) -> usize {
        self.get_window(current_time).len()
    }

    /// Clear old items
    fn cleanup(&mut self, current_time: DateTime<Utc>) {
        let window_start = current_time
            - chrono::Duration::seconds(self.window.as_seconds() as i64);

        self.data.retain(|(ts, _)| *ts >= window_start);
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get latest item
    pub fn latest(&self) -> Option<&T> {
        self.data.last().map(|(_, item)| item)
    }
}

/// Feature vector for ML model
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeatureVector {
    /// Feature names
    pub names: Vec<String>,
    /// Feature values
    pub values: Vec<f64>,
    /// Target label (if known)
    pub label: Option<f64>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl FeatureVector {
    /// Create empty feature vector
    pub fn new(timestamp: DateTime<Utc>) -> Self {
        Self {
            timestamp,
            ..Default::default()
        }
    }

    /// Add a feature
    pub fn add(&mut self, name: &str, value: f64) {
        self.names.push(name.to_string());
        self.values.push(value);
    }

    /// Set label
    pub fn set_label(&mut self, label: f64) {
        self.label = Some(label);
    }

    /// Get feature by name
    pub fn get(&self, name: &str) -> Option<f64> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| self.values[i])
    }

    /// Number of features
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Convert to array for ML
    pub fn to_array(&self) -> Vec<f64> {
        self.values.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_window() {
        assert_eq!(TimeWindow::Seconds(30).as_seconds(), 30);
        assert_eq!(TimeWindow::Minutes(5).as_seconds(), 300);
        assert_eq!(TimeWindow::Hours(1).as_seconds(), 3600);
    }

    #[test]
    fn test_feature_vector() {
        let mut fv = FeatureVector::new(Utc::now());
        fv.add("ofi", 1.5);
        fv.add("spread", 0.01);
        fv.set_label(1.0);

        assert_eq!(fv.len(), 2);
        assert!((fv.get("ofi").unwrap() - 1.5).abs() < 0.001);
        assert!(fv.label.is_some());
    }

    #[test]
    fn test_rolling_buffer() {
        let mut buffer: RollingBuffer<f64> = RollingBuffer::new(TimeWindow::Seconds(60), 1000);

        let now = Utc::now();
        buffer.push(now - chrono::Duration::seconds(30), 1.0);
        buffer.push(now - chrono::Duration::seconds(20), 2.0);
        buffer.push(now - chrono::Duration::seconds(10), 3.0);
        buffer.push(now, 4.0);

        assert_eq!(buffer.count(now), 4);
        assert!((buffer.latest().unwrap() - 4.0).abs() < 0.001);
    }
}
