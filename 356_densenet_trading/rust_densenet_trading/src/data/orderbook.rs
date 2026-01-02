//! Order Book Data Structure
//!
//! Represents the order book with bids and asks.

use serde::{Deserialize, Serialize};

/// A single level in the order book
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Price level
    pub price: f64,

    /// Quantity at this price
    pub quantity: f64,
}

impl OrderBookLevel {
    /// Create a new order book level
    pub fn new(price: f64, quantity: f64) -> Self {
        Self { price, quantity }
    }

    /// Get the total value at this level
    pub fn value(&self) -> f64 {
        self.price * self.quantity
    }
}

/// Order book with bids and asks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Symbol
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

    /// Get the spread as a percentage
    pub fn spread_percent(&self) -> Option<f64> {
        match (self.mid_price(), self.spread()) {
            (Some(mid), Some(spread)) if mid > 0.0 => Some(spread / mid * 100.0),
            _ => None,
        }
    }

    /// Get total bid volume
    pub fn total_bid_volume(&self) -> f64 {
        self.bids.iter().map(|l| l.quantity).sum()
    }

    /// Get total ask volume
    pub fn total_ask_volume(&self) -> f64 {
        self.asks.iter().map(|l| l.quantity).sum()
    }

    /// Get bid volume up to a certain depth (number of levels)
    pub fn bid_volume_at_depth(&self, depth: usize) -> f64 {
        self.bids.iter().take(depth).map(|l| l.quantity).sum()
    }

    /// Get ask volume up to a certain depth
    pub fn ask_volume_at_depth(&self, depth: usize) -> f64 {
        self.asks.iter().take(depth).map(|l| l.quantity).sum()
    }

    /// Get order imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
    pub fn order_imbalance(&self) -> f64 {
        let bid_vol = self.total_bid_volume();
        let ask_vol = self.total_ask_volume();
        let total = bid_vol + ask_vol;

        if total > 0.0 {
            (bid_vol - ask_vol) / total
        } else {
            0.0
        }
    }

    /// Get order imbalance at a specific depth
    pub fn order_imbalance_at_depth(&self, depth: usize) -> f64 {
        let bid_vol = self.bid_volume_at_depth(depth);
        let ask_vol = self.ask_volume_at_depth(depth);
        let total = bid_vol + ask_vol;

        if total > 0.0 {
            (bid_vol - ask_vol) / total
        } else {
            0.0
        }
    }

    /// Get the weighted mid price (weighted by volumes at best levels)
    pub fn weighted_mid_price(&self) -> Option<f64> {
        if let (Some(bid), Some(ask)) = (self.bids.first(), self.asks.first()) {
            let total_qty = bid.quantity + ask.quantity;
            if total_qty > 0.0 {
                Some((bid.price * ask.quantity + ask.price * bid.quantity) / total_qty)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Calculate VWAP for given quantity on bid side
    pub fn bid_vwap(&self, quantity: f64) -> Option<f64> {
        self.calculate_vwap(&self.bids, quantity)
    }

    /// Calculate VWAP for given quantity on ask side
    pub fn ask_vwap(&self, quantity: f64) -> Option<f64> {
        self.calculate_vwap(&self.asks, quantity)
    }

    fn calculate_vwap(&self, levels: &[OrderBookLevel], quantity: f64) -> Option<f64> {
        if levels.is_empty() || quantity <= 0.0 {
            return None;
        }

        let mut remaining = quantity;
        let mut total_value = 0.0;
        let mut total_qty = 0.0;

        for level in levels {
            let fill_qty = remaining.min(level.quantity);
            total_value += level.price * fill_qty;
            total_qty += fill_qty;
            remaining -= fill_qty;

            if remaining <= 0.0 {
                break;
            }
        }

        if total_qty > 0.0 {
            Some(total_value / total_qty)
        } else {
            None
        }
    }

    /// Get depth imbalance across multiple levels
    pub fn depth_imbalance(&self, levels: usize) -> Vec<f64> {
        (1..=levels)
            .map(|d| self.order_imbalance_at_depth(d))
            .collect()
    }

    /// Get price levels for bids
    pub fn bid_prices(&self) -> Vec<f64> {
        self.bids.iter().map(|l| l.price).collect()
    }

    /// Get price levels for asks
    pub fn ask_prices(&self) -> Vec<f64> {
        self.asks.iter().map(|l| l.price).collect()
    }

    /// Get bid quantities
    pub fn bid_quantities(&self) -> Vec<f64> {
        self.bids.iter().map(|l| l.quantity).collect()
    }

    /// Get ask quantities
    pub fn ask_quantities(&self) -> Vec<f64> {
        self.asks.iter().map(|l| l.quantity).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_orderbook() -> OrderBook {
        OrderBook::new(
            "BTCUSDT".to_string(),
            vec![
                OrderBookLevel::new(42000.0, 1.5),
                OrderBookLevel::new(41999.0, 2.0),
                OrderBookLevel::new(41998.0, 3.0),
            ],
            vec![
                OrderBookLevel::new(42001.0, 1.0),
                OrderBookLevel::new(42002.0, 2.5),
                OrderBookLevel::new(42003.0, 1.5),
            ],
            1704067200000,
        )
    }

    #[test]
    fn test_best_prices() {
        let ob = sample_orderbook();
        assert_eq!(ob.best_bid(), Some(42000.0));
        assert_eq!(ob.best_ask(), Some(42001.0));
    }

    #[test]
    fn test_mid_price() {
        let ob = sample_orderbook();
        assert_eq!(ob.mid_price(), Some(42000.5));
    }

    #[test]
    fn test_spread() {
        let ob = sample_orderbook();
        assert_eq!(ob.spread(), Some(1.0));
    }

    #[test]
    fn test_volumes() {
        let ob = sample_orderbook();
        assert!((ob.total_bid_volume() - 6.5).abs() < 0.001);
        assert!((ob.total_ask_volume() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_order_imbalance() {
        let ob = sample_orderbook();
        let imb = ob.order_imbalance();
        // (6.5 - 5.0) / (6.5 + 5.0) = 1.5 / 11.5 ≈ 0.1304
        assert!((imb - 0.1304).abs() < 0.01);
    }

    #[test]
    fn test_weighted_mid() {
        let ob = sample_orderbook();
        let wmid = ob.weighted_mid_price().unwrap();
        // (42000 * 1.0 + 42001 * 1.5) / (1.5 + 1.0) = (42000 + 63001.5) / 2.5
        assert!(wmid > 42000.0 && wmid < 42001.0);
    }

    #[test]
    fn test_vwap() {
        let ob = sample_orderbook();

        // Buy 1.0 BTC at ask
        let vwap = ob.ask_vwap(1.0).unwrap();
        assert!((vwap - 42001.0).abs() < 0.001);

        // Buy 2.0 BTC at ask (spans 2 levels)
        let vwap2 = ob.ask_vwap(2.0).unwrap();
        assert!(vwap2 > 42001.0); // Should be higher due to slippage
    }
}
