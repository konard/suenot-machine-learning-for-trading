//! # Order Book Data Structures
//!
//! Structures for representing and analyzing order book data.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A single level in the order book
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Price at this level
    pub price: f64,
    /// Size (quantity) at this level
    pub size: f64,
    /// Level number (1 = best, 2 = second best, etc.)
    pub level: usize,
}

impl OrderBookLevel {
    /// Create a new order book level
    pub fn new(price: f64, size: f64, level: usize) -> Self {
        Self { price, size, level }
    }

    /// Get the notional value at this level
    pub fn notional(&self) -> f64 {
        self.price * self.size
    }
}

/// Complete order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Trading symbol
    pub symbol: String,
    /// Timestamp of the snapshot
    pub timestamp: DateTime<Utc>,
    /// Bid (buy) levels, sorted by price descending
    pub bids: Vec<OrderBookLevel>,
    /// Ask (sell) levels, sorted by price ascending
    pub asks: Vec<OrderBookLevel>,
}

impl OrderBook {
    /// Create a new order book
    pub fn new(
        symbol: String,
        timestamp: DateTime<Utc>,
        bids: Vec<OrderBookLevel>,
        asks: Vec<OrderBookLevel>,
    ) -> Self {
        Self {
            symbol,
            timestamp,
            bids,
            asks,
        }
    }

    /// Get the best bid (highest buy price)
    pub fn best_bid(&self) -> Option<&OrderBookLevel> {
        self.bids.first()
    }

    /// Get the best ask (lowest sell price)
    pub fn best_ask(&self) -> Option<&OrderBookLevel> {
        self.asks.first()
    }

    /// Get the mid price
    pub fn mid_price(&self) -> Option<f64> {
        let bid = self.best_bid()?.price;
        let ask = self.best_ask()?.price;
        Some((bid + ask) / 2.0)
    }

    /// Get the spread in price units
    pub fn spread(&self) -> Option<f64> {
        let bid = self.best_bid()?.price;
        let ask = self.best_ask()?.price;
        Some(ask - bid)
    }

    /// Get the spread in basis points
    pub fn spread_bps(&self) -> Option<f64> {
        let spread = self.spread()?;
        let mid = self.mid_price()?;
        Some(spread / mid * 10000.0)
    }

    /// Get total bid depth up to N levels
    pub fn bid_depth(&self, levels: usize) -> f64 {
        self.bids.iter().take(levels).map(|l| l.size).sum()
    }

    /// Get total ask depth up to N levels
    pub fn ask_depth(&self, levels: usize) -> f64 {
        self.asks.iter().take(levels).map(|l| l.size).sum()
    }

    /// Get total depth (bid + ask) up to N levels
    pub fn total_depth(&self, levels: usize) -> f64 {
        self.bid_depth(levels) + self.ask_depth(levels)
    }

    /// Get bid notional value up to N levels
    pub fn bid_notional(&self, levels: usize) -> f64 {
        self.bids.iter().take(levels).map(|l| l.notional()).sum()
    }

    /// Get ask notional value up to N levels
    pub fn ask_notional(&self, levels: usize) -> f64 {
        self.asks.iter().take(levels).map(|l| l.notional()).sum()
    }

    /// Calculate depth imbalance at level N
    ///
    /// Returns (bid_size - ask_size) / (bid_size + ask_size)
    /// Range: [-1, 1], positive = more bids
    pub fn depth_imbalance(&self, levels: usize) -> f64 {
        let bid = self.bid_depth(levels);
        let ask = self.ask_depth(levels);
        let total = bid + ask;

        if total > 0.0 {
            (bid - ask) / total
        } else {
            0.0
        }
    }

    /// Calculate weighted depth imbalance
    ///
    /// Weights each level by inverse of distance from mid
    pub fn weighted_depth_imbalance(&self, levels: usize) -> Option<f64> {
        let mid = self.mid_price()?;
        let mut weighted_bid = 0.0;
        let mut weighted_ask = 0.0;

        for level in self.bids.iter().take(levels) {
            let distance = (mid - level.price) / mid;
            let weight = 1.0 / (1.0 + distance * 100.0); // Decay factor
            weighted_bid += level.size * weight;
        }

        for level in self.asks.iter().take(levels) {
            let distance = (level.price - mid) / mid;
            let weight = 1.0 / (1.0 + distance * 100.0);
            weighted_ask += level.size * weight;
        }

        let total = weighted_bid + weighted_ask;
        if total > 0.0 {
            Some((weighted_bid - weighted_ask) / total)
        } else {
            Some(0.0)
        }
    }

    /// Calculate price impact for buying a given size
    ///
    /// Returns the weighted average execution price
    pub fn price_impact_buy(&self, size: f64) -> Option<f64> {
        let mut remaining = size;
        let mut total_cost = 0.0;
        let mut total_filled = 0.0;

        for level in &self.asks {
            if remaining <= 0.0 {
                break;
            }

            let fill = remaining.min(level.size);
            total_cost += fill * level.price;
            total_filled += fill;
            remaining -= fill;
        }

        if total_filled > 0.0 {
            Some(total_cost / total_filled)
        } else {
            None
        }
    }

    /// Calculate price impact for selling a given size
    pub fn price_impact_sell(&self, size: f64) -> Option<f64> {
        let mut remaining = size;
        let mut total_proceeds = 0.0;
        let mut total_filled = 0.0;

        for level in &self.bids {
            if remaining <= 0.0 {
                break;
            }

            let fill = remaining.min(level.size);
            total_proceeds += fill * level.price;
            total_filled += fill;
            remaining -= fill;
        }

        if total_filled > 0.0 {
            Some(total_proceeds / total_filled)
        } else {
            None
        }
    }

    /// Calculate bid slope (price impact per unit size)
    ///
    /// Measures how quickly price deteriorates as we buy/sell
    pub fn bid_slope(&self, levels: usize) -> Option<f64> {
        if self.bids.len() < 2 {
            return None;
        }

        let levels = levels.min(self.bids.len());
        let first = &self.bids[0];
        let last = &self.bids[levels - 1];

        let price_diff = first.price - last.price;
        let cumulative_size: f64 = self.bids.iter().take(levels).map(|l| l.size).sum();

        if cumulative_size > 0.0 {
            Some(price_diff / cumulative_size)
        } else {
            None
        }
    }

    /// Calculate ask slope
    pub fn ask_slope(&self, levels: usize) -> Option<f64> {
        if self.asks.len() < 2 {
            return None;
        }

        let levels = levels.min(self.asks.len());
        let first = &self.asks[0];
        let last = &self.asks[levels - 1];

        let price_diff = last.price - first.price;
        let cumulative_size: f64 = self.asks.iter().take(levels).map(|l| l.size).sum();

        if cumulative_size > 0.0 {
            Some(price_diff / cumulative_size)
        } else {
            None
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

    /// Check if the order book is valid (has at least bid and ask)
    pub fn is_valid(&self) -> bool {
        !self.bids.is_empty() && !self.asks.is_empty()
    }

    /// Check for crossed book (bid >= ask, invalid state)
    pub fn is_crossed(&self) -> bool {
        if let (Some(bid), Some(ask)) = (self.best_bid(), self.best_ask()) {
            bid.price >= ask.price
        } else {
            false
        }
    }
}

/// Order book update (delta)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookUpdate {
    pub timestamp: DateTime<Utc>,
    pub side: Side,
    pub price: f64,
    pub size: f64, // 0 means remove
}

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Bid,
    Ask,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_orderbook() -> OrderBook {
        let bids = vec![
            OrderBookLevel::new(100.0, 10.0, 1),
            OrderBookLevel::new(99.5, 20.0, 2),
            OrderBookLevel::new(99.0, 15.0, 3),
        ];

        let asks = vec![
            OrderBookLevel::new(100.5, 8.0, 1),
            OrderBookLevel::new(101.0, 25.0, 2),
            OrderBookLevel::new(101.5, 12.0, 3),
        ];

        OrderBook::new("BTCUSDT".to_string(), Utc::now(), bids, asks)
    }

    #[test]
    fn test_mid_price() {
        let ob = sample_orderbook();
        let mid = ob.mid_price().unwrap();
        assert!((mid - 100.25).abs() < 0.001);
    }

    #[test]
    fn test_spread() {
        let ob = sample_orderbook();
        let spread = ob.spread().unwrap();
        assert!((spread - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_spread_bps() {
        let ob = sample_orderbook();
        let spread_bps = ob.spread_bps().unwrap();
        // 0.5 / 100.25 * 10000 ≈ 49.88 bps
        assert!(spread_bps > 49.0 && spread_bps < 51.0);
    }

    #[test]
    fn test_depth() {
        let ob = sample_orderbook();

        assert!((ob.bid_depth(1) - 10.0).abs() < 0.001);
        assert!((ob.bid_depth(3) - 45.0).abs() < 0.001);

        assert!((ob.ask_depth(1) - 8.0).abs() < 0.001);
        assert!((ob.ask_depth(3) - 45.0).abs() < 0.001);
    }

    #[test]
    fn test_depth_imbalance() {
        let ob = sample_orderbook();
        let imbalance = ob.depth_imbalance(1);
        // (10 - 8) / (10 + 8) = 2/18 ≈ 0.111
        assert!((imbalance - 0.111).abs() < 0.01);
    }

    #[test]
    fn test_is_valid() {
        let ob = sample_orderbook();
        assert!(ob.is_valid());
        assert!(!ob.is_crossed());
    }

    #[test]
    fn test_price_impact() {
        let ob = sample_orderbook();

        // Buy 5 BTC - should fill at 100.5
        let impact = ob.price_impact_buy(5.0).unwrap();
        assert!((impact - 100.5).abs() < 0.001);

        // Buy 10 BTC - should fill across two levels
        let impact = ob.price_impact_buy(10.0).unwrap();
        // 8 @ 100.5 + 2 @ 101.0 = 805 + 202 = 1007 / 10 = 100.7
        assert!((impact - 100.7).abs() < 0.001);
    }
}
