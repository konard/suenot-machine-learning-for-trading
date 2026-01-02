//! Order book data structures and processing

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Single order book level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Price level
    pub price: f64,
    /// Size at this price
    pub size: f64,
}

impl OrderBookLevel {
    pub fn new(price: f64, size: f64) -> Self {
        Self { price, size }
    }

    /// Get notional value
    pub fn notional(&self) -> f64 {
        self.price * self.size
    }
}

/// Order book snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    /// Best bid price
    pub best_bid: f64,
    /// Best ask price
    pub best_ask: f64,
    /// Mid price
    pub mid_price: f64,
    /// Spread
    pub spread: f64,
    /// Spread percentage
    pub spread_pct: f64,
    /// Bid depth (total size within X%)
    pub bid_depth: f64,
    /// Ask depth
    pub ask_depth: f64,
    /// Imbalance (-1 to 1)
    pub imbalance: f64,
    /// Weighted mid price
    pub weighted_mid: f64,
    /// Timestamp
    pub timestamp: u64,
}

/// Full order book with bid and ask sides
#[derive(Debug, Clone)]
pub struct OrderBook {
    /// Symbol
    pub symbol: String,
    /// Bid levels (price -> size), sorted descending by price
    bids: BTreeMap<OrderedFloat, f64>,
    /// Ask levels (price -> size), sorted ascending by price
    asks: BTreeMap<OrderedFloat, f64>,
    /// Last update timestamp
    pub timestamp: u64,
}

/// Wrapper for f64 to enable ordering
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedFloat(f64);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl OrderBook {
    /// Create a new order book
    pub fn new(symbol: impl Into<String>, bids: Vec<OrderBookLevel>, asks: Vec<OrderBookLevel>) -> Self {
        let mut book = Self {
            symbol: symbol.into(),
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };

        for level in bids {
            book.bids.insert(OrderedFloat(level.price), level.size);
        }
        for level in asks {
            book.asks.insert(OrderedFloat(level.price), level.size);
        }

        book
    }

    /// Create empty order book
    pub fn empty(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            timestamp: 0,
        }
    }

    /// Get best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.keys().next_back().map(|k| k.0)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.keys().next().map(|k| k.0)
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

    /// Get spread as percentage of mid
    pub fn spread_pct(&self) -> Option<f64> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid)) if mid > 0.0 => Some(spread / mid),
            _ => None,
        }
    }

    /// Get best bid size
    pub fn best_bid_size(&self) -> Option<f64> {
        self.bids.values().next_back().copied()
    }

    /// Get best ask size
    pub fn best_ask_size(&self) -> Option<f64> {
        self.asks.values().next().copied()
    }

    /// Get bid depth within percentage of mid price
    pub fn bid_depth(&self, pct: f64) -> f64 {
        let mid = self.mid_price().unwrap_or(0.0);
        let threshold = mid * (1.0 - pct);

        self.bids
            .iter()
            .filter(|(price, _)| price.0 >= threshold)
            .map(|(_, size)| size)
            .sum()
    }

    /// Get ask depth within percentage of mid price
    pub fn ask_depth(&self, pct: f64) -> f64 {
        let mid = self.mid_price().unwrap_or(0.0);
        let threshold = mid * (1.0 + pct);

        self.asks
            .iter()
            .filter(|(price, _)| price.0 <= threshold)
            .map(|(_, size)| size)
            .sum()
    }

    /// Calculate order book imbalance
    pub fn imbalance(&self, levels: usize) -> f64 {
        let bid_volume: f64 = self.bids.values().rev().take(levels).sum();
        let ask_volume: f64 = self.asks.values().take(levels).sum();

        let total = bid_volume + ask_volume;
        if total > 0.0 {
            (bid_volume - ask_volume) / total
        } else {
            0.0
        }
    }

    /// Calculate weighted mid price based on top-of-book sizes
    pub fn weighted_mid(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask(), self.best_bid_size(), self.best_ask_size()) {
            (Some(bid), Some(ask), Some(bid_size), Some(ask_size)) => {
                let total = bid_size + ask_size;
                if total > 0.0 {
                    Some((bid * ask_size + ask * bid_size) / total)
                } else {
                    Some((bid + ask) / 2.0)
                }
            }
            _ => None,
        }
    }

    /// Get all bid levels as vector
    pub fn bid_levels(&self) -> Vec<OrderBookLevel> {
        self.bids
            .iter()
            .rev()
            .map(|(price, size)| OrderBookLevel::new(price.0, *size))
            .collect()
    }

    /// Get all ask levels as vector
    pub fn ask_levels(&self) -> Vec<OrderBookLevel> {
        self.asks
            .iter()
            .map(|(price, size)| OrderBookLevel::new(price.0, *size))
            .collect()
    }

    /// Calculate volume-weighted average price for a given size
    pub fn vwap_for_size(&self, size: f64, side: &str) -> Option<f64> {
        let levels = if side == "buy" {
            self.ask_levels()
        } else {
            self.bid_levels()
        };

        let mut remaining = size;
        let mut total_cost = 0.0;
        let mut total_filled = 0.0;

        for level in levels {
            if remaining <= 0.0 {
                break;
            }

            let fill_size = remaining.min(level.size);
            total_cost += fill_size * level.price;
            total_filled += fill_size;
            remaining -= fill_size;
        }

        if total_filled > 0.0 {
            Some(total_cost / total_filled)
        } else {
            None
        }
    }

    /// Calculate slippage for a given size
    pub fn slippage(&self, size: f64, side: &str) -> Option<f64> {
        let vwap = self.vwap_for_size(size, side)?;
        let mid = self.mid_price()?;

        if side == "buy" {
            Some((vwap - mid) / mid)
        } else {
            Some((mid - vwap) / mid)
        }
    }

    /// Take a snapshot of current state
    pub fn snapshot(&self) -> OrderBookSnapshot {
        let best_bid = self.best_bid().unwrap_or(0.0);
        let best_ask = self.best_ask().unwrap_or(0.0);
        let mid_price = self.mid_price().unwrap_or(0.0);
        let spread = self.spread().unwrap_or(0.0);

        OrderBookSnapshot {
            best_bid,
            best_ask,
            mid_price,
            spread,
            spread_pct: if mid_price > 0.0 { spread / mid_price } else { 0.0 },
            bid_depth: self.bid_depth(0.01), // 1% depth
            ask_depth: self.ask_depth(0.01),
            imbalance: self.imbalance(10),
            weighted_mid: self.weighted_mid().unwrap_or(mid_price),
            timestamp: self.timestamp,
        }
    }

    /// Update a bid level
    pub fn update_bid(&mut self, price: f64, size: f64) {
        if size > 0.0 {
            self.bids.insert(OrderedFloat(price), size);
        } else {
            self.bids.remove(&OrderedFloat(price));
        }
    }

    /// Update an ask level
    pub fn update_ask(&mut self, price: f64, size: f64) {
        if size > 0.0 {
            self.asks.insert(OrderedFloat(price), size);
        } else {
            self.asks.remove(&OrderedFloat(price));
        }
    }

    /// Get number of bid levels
    pub fn bid_level_count(&self) -> usize {
        self.bids.len()
    }

    /// Get number of ask levels
    pub fn ask_level_count(&self) -> usize {
        self.asks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_book() -> OrderBook {
        let bids = vec![
            OrderBookLevel::new(100.0, 10.0),
            OrderBookLevel::new(99.0, 20.0),
            OrderBookLevel::new(98.0, 30.0),
        ];
        let asks = vec![
            OrderBookLevel::new(101.0, 10.0),
            OrderBookLevel::new(102.0, 20.0),
            OrderBookLevel::new(103.0, 30.0),
        ];
        OrderBook::new("TEST", bids, asks)
    }

    #[test]
    fn test_best_prices() {
        let book = create_test_book();
        assert_eq!(book.best_bid(), Some(100.0));
        assert_eq!(book.best_ask(), Some(101.0));
    }

    #[test]
    fn test_mid_price() {
        let book = create_test_book();
        assert_eq!(book.mid_price(), Some(100.5));
    }

    #[test]
    fn test_spread() {
        let book = create_test_book();
        assert_eq!(book.spread(), Some(1.0));
    }

    #[test]
    fn test_imbalance() {
        let book = create_test_book();
        let imbalance = book.imbalance(3);
        assert_eq!(imbalance, 0.0); // Equal volume on both sides
    }

    #[test]
    fn test_vwap() {
        let book = create_test_book();

        // Buy 15 units: 10 @ 101 + 5 @ 102 = 1010 + 510 = 1520 / 15 = 101.33
        let vwap = book.vwap_for_size(15.0, "buy").unwrap();
        assert!((vwap - 101.333).abs() < 0.01);
    }

    #[test]
    fn test_weighted_mid() {
        let book = create_test_book();
        let wmid = book.weighted_mid().unwrap();
        // Equal sizes at best, so should equal mid
        assert_eq!(wmid, 100.5);
    }

    #[test]
    fn test_snapshot() {
        let book = create_test_book();
        let snapshot = book.snapshot();

        assert_eq!(snapshot.best_bid, 100.0);
        assert_eq!(snapshot.best_ask, 101.0);
        assert_eq!(snapshot.mid_price, 100.5);
    }
}
