//! Order Book Data Structure
//!
//! Represents the order book with bids and asks for market microstructure analysis.

use serde::{Deserialize, Serialize};

/// A single level in the order book
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrderBookLevel {
    /// Price at this level
    pub price: f64,

    /// Quantity at this level
    pub quantity: f64,
}

impl OrderBookLevel {
    /// Create a new order book level
    pub fn new(price: f64, quantity: f64) -> Self {
        Self { price, quantity }
    }

    /// Calculate the notional value at this level
    pub fn notional(&self) -> f64 {
        self.price * self.quantity
    }
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Trading symbol
    pub symbol: String,

    /// Bid levels (sorted by price descending)
    pub bids: Vec<OrderBookLevel>,

    /// Ask levels (sorted by price ascending)
    pub asks: Vec<OrderBookLevel>,

    /// Timestamp in milliseconds
    pub timestamp: u64,
}

impl OrderBook {
    /// Create a new order book
    pub fn new(
        symbol: String,
        bids: Vec<OrderBookLevel>,
        asks: Vec<OrderBookLevel>,
        timestamp: u64,
    ) -> Self {
        Self {
            symbol,
            bids,
            asks,
            timestamp,
        }
    }

    /// Get the best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Get the best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Calculate the mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Calculate the bid-ask spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Calculate the spread as a percentage of mid price
    pub fn spread_pct(&self) -> Option<f64> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid)) if mid > 0.0 => Some(spread / mid),
            _ => None,
        }
    }

    /// Calculate total bid volume up to a certain depth
    pub fn bid_volume(&self, depth: usize) -> f64 {
        self.bids.iter().take(depth).map(|l| l.quantity).sum()
    }

    /// Calculate total ask volume up to a certain depth
    pub fn ask_volume(&self, depth: usize) -> f64 {
        self.asks.iter().take(depth).map(|l| l.quantity).sum()
    }

    /// Calculate order imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
    pub fn order_imbalance(&self, depth: usize) -> f64 {
        let bid_vol = self.bid_volume(depth);
        let ask_vol = self.ask_volume(depth);
        let total = bid_vol + ask_vol;

        if total.abs() < 1e-10 {
            0.0
        } else {
            (bid_vol - ask_vol) / total
        }
    }

    /// Calculate depth imbalance at multiple levels
    pub fn depth_imbalance(&self, levels: &[usize]) -> Vec<f64> {
        levels.iter().map(|&d| self.order_imbalance(d)).collect()
    }

    /// Calculate total bid notional value
    pub fn bid_notional(&self, depth: usize) -> f64 {
        self.bids.iter().take(depth).map(|l| l.notional()).sum()
    }

    /// Calculate total ask notional value
    pub fn ask_notional(&self, depth: usize) -> f64 {
        self.asks.iter().take(depth).map(|l| l.notional()).sum()
    }

    /// Calculate VWAP of bids
    pub fn bid_vwap(&self, depth: usize) -> Option<f64> {
        let notional: f64 = self.bids.iter().take(depth).map(|l| l.notional()).sum();
        let volume: f64 = self.bids.iter().take(depth).map(|l| l.quantity).sum();

        if volume.abs() < 1e-10 {
            None
        } else {
            Some(notional / volume)
        }
    }

    /// Calculate VWAP of asks
    pub fn ask_vwap(&self, depth: usize) -> Option<f64> {
        let notional: f64 = self.asks.iter().take(depth).map(|l| l.notional()).sum();
        let volume: f64 = self.asks.iter().take(depth).map(|l| l.quantity).sum();

        if volume.abs() < 1e-10 {
            None
        } else {
            Some(notional / volume)
        }
    }

    /// Calculate microprice (volume-weighted mid)
    pub fn microprice(&self) -> Option<f64> {
        if self.bids.is_empty() || self.asks.is_empty() {
            return None;
        }

        let bid = &self.bids[0];
        let ask = &self.asks[0];
        let total_qty = bid.quantity + ask.quantity;

        if total_qty.abs() < 1e-10 {
            self.mid_price()
        } else {
            Some((bid.price * ask.quantity + ask.price * bid.quantity) / total_qty)
        }
    }

    /// Extract features for ML models
    pub fn extract_features(&self, depth: usize) -> Vec<f64> {
        let mut features = Vec::new();

        // Spread features
        features.push(self.spread_pct().unwrap_or(0.0));

        // Imbalance at different depths
        for d in [1, 5, 10, 20].iter() {
            let d = (*d).min(depth);
            features.push(self.order_imbalance(d));
        }

        // Volume features
        let bid_vol = self.bid_volume(depth);
        let ask_vol = self.ask_volume(depth);
        let total_vol = bid_vol + ask_vol;

        features.push(if total_vol > 0.0 {
            bid_vol / total_vol
        } else {
            0.5
        });

        // Microprice deviation from mid
        if let (Some(micro), Some(mid)) = (self.microprice(), self.mid_price()) {
            features.push((micro - mid) / mid);
        } else {
            features.push(0.0);
        }

        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_orderbook() -> OrderBook {
        let bids = vec![
            OrderBookLevel::new(100.0, 10.0),
            OrderBookLevel::new(99.0, 20.0),
            OrderBookLevel::new(98.0, 30.0),
        ];
        let asks = vec![
            OrderBookLevel::new(101.0, 15.0),
            OrderBookLevel::new(102.0, 25.0),
            OrderBookLevel::new(103.0, 35.0),
        ];

        OrderBook::new("BTCUSDT".to_string(), bids, asks, 1700000000000)
    }

    #[test]
    fn test_mid_price() {
        let ob = create_test_orderbook();
        let mid = ob.mid_price().unwrap();
        assert!((mid - 100.5).abs() < 1e-10);
    }

    #[test]
    fn test_spread() {
        let ob = create_test_orderbook();
        let spread = ob.spread().unwrap();
        assert!((spread - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_order_imbalance() {
        let ob = create_test_orderbook();

        // At depth 1: bid=10, ask=15, imbalance = (10-15)/(10+15) = -0.2
        let imb = ob.order_imbalance(1);
        assert!((imb - (-0.2)).abs() < 1e-10);
    }

    #[test]
    fn test_microprice() {
        let ob = create_test_orderbook();
        let micro = ob.microprice().unwrap();

        // microprice = (100*15 + 101*10) / (10+15) = 2510/25 = 100.4
        assert!((micro - 100.4).abs() < 1e-10);
    }

    #[test]
    fn test_extract_features() {
        let ob = create_test_orderbook();
        let features = ob.extract_features(20);
        assert!(!features.is_empty());
    }
}
