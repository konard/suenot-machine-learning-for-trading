//! Order book data structures and operations.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Single level in the order book
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Price level
    pub price: f64,
    /// Quantity at this level
    pub quantity: f64,
}

impl OrderBookLevel {
    /// Create a new order book level
    pub fn new(price: f64, quantity: f64) -> Self {
        Self { price, quantity }
    }

    /// Get the value (price * quantity)
    pub fn value(&self) -> f64 {
        self.price * self.quantity
    }
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    /// Symbol
    pub symbol: String,
    /// Bid levels (sorted by price descending)
    pub bids: Vec<OrderBookLevel>,
    /// Ask levels (sorted by price ascending)
    pub asks: Vec<OrderBookLevel>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Sequence number (for delta updates)
    pub sequence: Option<u64>,
}

/// Full order book with efficient updates
#[derive(Debug, Clone)]
pub struct OrderBook {
    /// Symbol
    pub symbol: String,
    /// Bid side: price -> quantity (sorted descending)
    bids: BTreeMap<OrderedFloat, f64>,
    /// Ask side: price -> quantity (sorted ascending)
    asks: BTreeMap<OrderedFloat, f64>,
    /// Last update timestamp
    pub last_update: DateTime<Utc>,
    /// Sequence number
    pub sequence: u64,
}

/// Wrapper for f64 to implement Ord for BTreeMap
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
    /// Create a new empty order book
    pub fn new(symbol: String) -> Self {
        Self {
            symbol,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_update: Utc::now(),
            sequence: 0,
        }
    }

    /// Initialize from snapshot
    pub fn from_snapshot(snapshot: OrderBookSnapshot) -> Self {
        let mut book = Self::new(snapshot.symbol);
        book.last_update = snapshot.timestamp;
        book.sequence = snapshot.sequence.unwrap_or(0);

        for level in snapshot.bids {
            book.bids.insert(OrderedFloat(level.price), level.quantity);
        }
        for level in snapshot.asks {
            book.asks.insert(OrderedFloat(level.price), level.quantity);
        }

        book
    }

    /// Update a bid level (quantity = 0 removes the level)
    pub fn update_bid(&mut self, price: f64, quantity: f64) {
        let key = OrderedFloat(price);
        if quantity <= 0.0 {
            self.bids.remove(&key);
        } else {
            self.bids.insert(key, quantity);
        }
        self.last_update = Utc::now();
    }

    /// Update an ask level (quantity = 0 removes the level)
    pub fn update_ask(&mut self, price: f64, quantity: f64) {
        let key = OrderedFloat(price);
        if quantity <= 0.0 {
            self.asks.remove(&key);
        } else {
            self.asks.insert(key, quantity);
        }
        self.last_update = Utc::now();
    }

    /// Get the best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.iter().next_back().map(|(k, _)| k.0)
    }

    /// Get the best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.iter().next().map(|(k, _)| k.0)
    }

    /// Get the best bid with quantity
    pub fn best_bid_level(&self) -> Option<OrderBookLevel> {
        self.bids
            .iter()
            .next_back()
            .map(|(k, v)| OrderBookLevel::new(k.0, *v))
    }

    /// Get the best ask with quantity
    pub fn best_ask_level(&self) -> Option<OrderBookLevel> {
        self.asks
            .iter()
            .next()
            .map(|(k, v)| OrderBookLevel::new(k.0, *v))
    }

    /// Get the mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Get the spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get the spread in basis points
    pub fn spread_bps(&self) -> Option<f64> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid)) if mid > 0.0 => Some((spread / mid) * 10000.0),
            _ => None,
        }
    }

    /// Get top N bid levels
    pub fn top_bids(&self, n: usize) -> Vec<OrderBookLevel> {
        self.bids
            .iter()
            .rev()
            .take(n)
            .map(|(k, v)| OrderBookLevel::new(k.0, *v))
            .collect()
    }

    /// Get top N ask levels
    pub fn top_asks(&self, n: usize) -> Vec<OrderBookLevel> {
        self.asks
            .iter()
            .take(n)
            .map(|(k, v)| OrderBookLevel::new(k.0, *v))
            .collect()
    }

    /// Calculate total bid depth up to a price level
    pub fn bid_depth_to_price(&self, price: f64) -> f64 {
        self.bids
            .iter()
            .rev()
            .take_while(|(k, _)| k.0 >= price)
            .map(|(_, v)| v)
            .sum()
    }

    /// Calculate total ask depth up to a price level
    pub fn ask_depth_to_price(&self, price: f64) -> f64 {
        self.asks
            .iter()
            .take_while(|(k, _)| k.0 <= price)
            .map(|(_, v)| v)
            .sum()
    }

    /// Calculate total bid depth for N levels
    pub fn bid_depth(&self, levels: usize) -> f64 {
        self.bids.iter().rev().take(levels).map(|(_, v)| v).sum()
    }

    /// Calculate total ask depth for N levels
    pub fn ask_depth(&self, levels: usize) -> f64 {
        self.asks.iter().take(levels).map(|(_, v)| v).sum()
    }

    /// Calculate the imbalance ratio (bid_depth - ask_depth) / (bid_depth + ask_depth)
    pub fn imbalance(&self, levels: usize) -> f64 {
        let bid_depth = self.bid_depth(levels);
        let ask_depth = self.ask_depth(levels);
        let total = bid_depth + ask_depth;
        if total > 0.0 {
            (bid_depth - ask_depth) / total
        } else {
            0.0
        }
    }

    /// Estimate the price impact of buying a given quantity
    pub fn buy_impact(&self, quantity: f64) -> Option<(f64, f64)> {
        self.estimate_impact(&self.top_asks(100), quantity)
    }

    /// Estimate the price impact of selling a given quantity
    pub fn sell_impact(&self, quantity: f64) -> Option<(f64, f64)> {
        self.estimate_impact(&self.top_bids(100), quantity)
    }

    /// Estimate impact from a list of levels
    fn estimate_impact(&self, levels: &[OrderBookLevel], quantity: f64) -> Option<(f64, f64)> {
        if levels.is_empty() || quantity <= 0.0 {
            return None;
        }

        let mut remaining = quantity;
        let mut total_cost = 0.0;
        let mut last_price = levels[0].price;

        for level in levels {
            if remaining <= 0.0 {
                break;
            }

            let fill_qty = remaining.min(level.quantity);
            total_cost += fill_qty * level.price;
            remaining -= fill_qty;
            last_price = level.price;
        }

        if remaining > 0.0 {
            // Not enough liquidity
            return None;
        }

        let avg_price = total_cost / quantity;
        let impact = (last_price - levels[0].price).abs() / levels[0].price;

        Some((avg_price, impact))
    }

    /// Get a snapshot of the order book
    pub fn to_snapshot(&self) -> OrderBookSnapshot {
        OrderBookSnapshot {
            symbol: self.symbol.clone(),
            bids: self.top_bids(50),
            asks: self.top_asks(50),
            timestamp: self.last_update,
            sequence: Some(self.sequence),
        }
    }

    /// Get the number of bid levels
    pub fn bid_levels(&self) -> usize {
        self.bids.len()
    }

    /// Get the number of ask levels
    pub fn ask_levels(&self) -> usize {
        self.asks.len()
    }

    /// Clear all levels
    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.sequence = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_book() -> OrderBook {
        let mut book = OrderBook::new("BTCUSDT".to_string());

        // Add bids
        book.update_bid(49990.0, 1.0);
        book.update_bid(49980.0, 2.0);
        book.update_bid(49970.0, 3.0);

        // Add asks
        book.update_ask(50010.0, 1.0);
        book.update_ask(50020.0, 2.0);
        book.update_ask(50030.0, 3.0);

        book
    }

    #[test]
    fn test_best_prices() {
        let book = create_test_book();
        assert_eq!(book.best_bid(), Some(49990.0));
        assert_eq!(book.best_ask(), Some(50010.0));
    }

    #[test]
    fn test_mid_price_and_spread() {
        let book = create_test_book();
        assert_eq!(book.mid_price(), Some(50000.0));
        assert_eq!(book.spread(), Some(20.0));
        assert!((book.spread_bps().unwrap() - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_depth() {
        let book = create_test_book();
        assert_eq!(book.bid_depth(2), 3.0); // 1 + 2
        assert_eq!(book.ask_depth(2), 3.0); // 1 + 2
        assert_eq!(book.bid_depth(10), 6.0); // all: 1 + 2 + 3
    }

    #[test]
    fn test_imbalance() {
        let book = create_test_book();
        let imbalance = book.imbalance(3);
        assert!((imbalance - 0.0).abs() < 0.001); // Equal depth on both sides
    }

    #[test]
    fn test_buy_impact() {
        let book = create_test_book();
        let (avg_price, impact) = book.buy_impact(1.5).unwrap();

        // Should fill 1.0 at 50010 and 0.5 at 50020
        let expected_avg = (1.0 * 50010.0 + 0.5 * 50020.0) / 1.5;
        assert!((avg_price - expected_avg).abs() < 0.01);
        assert!(impact > 0.0);
    }

    #[test]
    fn test_update_remove_level() {
        let mut book = create_test_book();
        book.update_bid(49990.0, 0.0); // Remove best bid
        assert_eq!(book.best_bid(), Some(49980.0));
    }
}
