//! Order Book Implementation
//!
//! A limit order book with price-time priority matching algorithm.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use uuid::Uuid;

/// Order side (buy or sell)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    /// Market order - executes immediately at best available price
    Market,
    /// Limit order - executes only at specified price or better
    Limit,
}

/// A single order in the order book
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Unique order ID
    pub id: String,
    /// Agent who placed the order
    pub agent_id: String,
    /// Order side
    pub side: Side,
    /// Order type
    pub order_type: OrderType,
    /// Limit price (for limit orders)
    pub price: f64,
    /// Order quantity
    pub quantity: u64,
    /// Remaining quantity
    pub remaining: u64,
    /// Order timestamp
    pub timestamp: DateTime<Utc>,
}

impl Order {
    /// Create a new order
    pub fn new(
        agent_id: String,
        side: Side,
        order_type: OrderType,
        price: f64,
        quantity: u64,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            agent_id,
            side,
            order_type,
            price,
            quantity,
            remaining: quantity,
            timestamp: Utc::now(),
        }
    }

    /// Check if order is fully filled
    pub fn is_filled(&self) -> bool {
        self.remaining == 0
    }
}

// For buy orders: higher price = higher priority
// For sell orders: lower price = higher priority
// At same price: earlier timestamp = higher priority

#[derive(Debug, Clone)]
struct BidEntry(Order);

#[derive(Debug, Clone)]
struct AskEntry(Order);

impl PartialEq for BidEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0.id == other.0.id
    }
}

impl Eq for BidEntry {}

impl PartialOrd for BidEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BidEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher price first, then earlier time
        match self.0.price.partial_cmp(&other.0.price) {
            Some(Ordering::Equal) => other.0.timestamp.cmp(&self.0.timestamp),
            Some(ord) => ord,
            None => Ordering::Equal,
        }
    }
}

impl PartialEq for AskEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0.id == other.0.id
    }
}

impl Eq for AskEntry {}

impl PartialOrd for AskEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AskEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower price first, then earlier time
        match other.0.price.partial_cmp(&self.0.price) {
            Some(Ordering::Equal) => other.0.timestamp.cmp(&self.0.timestamp),
            Some(ord) => ord,
            None => Ordering::Equal,
        }
    }
}

/// Result of order execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderResult {
    /// Order ID
    pub order_id: String,
    /// Whether order was accepted
    pub accepted: bool,
    /// Filled quantity
    pub filled_quantity: u64,
    /// Average fill price
    pub average_price: f64,
    /// List of fills
    pub fills: Vec<Fill>,
    /// Remaining quantity in book
    pub remaining: u64,
}

/// A single fill (trade execution)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    /// Price at which trade occurred
    pub price: f64,
    /// Quantity traded
    pub quantity: u64,
    /// Counterparty agent ID
    pub counterparty: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Limit Order Book
#[derive(Debug)]
pub struct OrderBook {
    /// Buy orders (bids)
    bids: BinaryHeap<BidEntry>,
    /// Sell orders (asks)
    asks: BinaryHeap<AskEntry>,
    /// Last traded price
    pub last_price: f64,
    /// Trade history
    trades: Vec<Fill>,
}

impl OrderBook {
    /// Create a new empty order book
    pub fn new(initial_price: f64) -> Self {
        Self {
            bids: BinaryHeap::new(),
            asks: BinaryHeap::new(),
            last_price: initial_price,
            trades: Vec::new(),
        }
    }

    /// Get best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.peek().map(|e| e.0.price)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.peek().map(|e| e.0.price)
    }

    /// Get bid-ask spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) => Some((ask + bid) / 2.0),
            _ => None,
        }
    }

    /// Submit an order to the book
    pub fn submit_order(&mut self, mut order: Order) -> OrderResult {
        let mut result = OrderResult {
            order_id: order.id.clone(),
            accepted: true,
            filled_quantity: 0,
            average_price: 0.0,
            fills: Vec::new(),
            remaining: order.remaining,
        };

        match order.side {
            Side::Buy => self.match_buy_order(&mut order, &mut result),
            Side::Sell => self.match_sell_order(&mut order, &mut result),
        }

        // Capture remaining before potentially moving order
        let remaining = order.remaining;

        // Add remaining quantity to book (for limit orders)
        if remaining > 0 && order.order_type == OrderType::Limit {
            match order.side {
                Side::Buy => self.bids.push(BidEntry(order)),
                Side::Sell => self.asks.push(AskEntry(order)),
            }
        }

        // Calculate average price
        if result.filled_quantity > 0 {
            let total_value: f64 = result.fills.iter()
                .map(|f| f.price * f.quantity as f64)
                .sum();
            result.average_price = total_value / result.filled_quantity as f64;
            self.last_price = result.fills.last().map(|f| f.price).unwrap_or(self.last_price);
        }

        result.remaining = remaining;
        result
    }

    fn match_buy_order(&mut self, order: &mut Order, result: &mut OrderResult) {
        let mut matched_asks = Vec::new();

        while order.remaining > 0 {
            if let Some(mut ask_entry) = self.asks.pop() {
                let can_match = match order.order_type {
                    OrderType::Market => true,
                    OrderType::Limit => order.price >= ask_entry.0.price,
                };

                if can_match {
                    let fill_qty = order.remaining.min(ask_entry.0.remaining);
                    let fill_price = ask_entry.0.price;

                    let fill = Fill {
                        price: fill_price,
                        quantity: fill_qty,
                        counterparty: ask_entry.0.agent_id.clone(),
                        timestamp: Utc::now(),
                    };

                    order.remaining -= fill_qty;
                    ask_entry.0.remaining -= fill_qty;
                    result.filled_quantity += fill_qty;
                    result.fills.push(fill.clone());
                    self.trades.push(fill);

                    if ask_entry.0.remaining > 0 {
                        matched_asks.push(ask_entry);
                    }
                } else {
                    matched_asks.push(ask_entry);
                    break;
                }
            } else {
                break;
            }
        }

        // Put back unmatched asks
        for ask in matched_asks {
            self.asks.push(ask);
        }
    }

    fn match_sell_order(&mut self, order: &mut Order, result: &mut OrderResult) {
        let mut matched_bids = Vec::new();

        while order.remaining > 0 {
            if let Some(mut bid_entry) = self.bids.pop() {
                let can_match = match order.order_type {
                    OrderType::Market => true,
                    OrderType::Limit => order.price <= bid_entry.0.price,
                };

                if can_match {
                    let fill_qty = order.remaining.min(bid_entry.0.remaining);
                    let fill_price = bid_entry.0.price;

                    let fill = Fill {
                        price: fill_price,
                        quantity: fill_qty,
                        counterparty: bid_entry.0.agent_id.clone(),
                        timestamp: Utc::now(),
                    };

                    order.remaining -= fill_qty;
                    bid_entry.0.remaining -= fill_qty;
                    result.filled_quantity += fill_qty;
                    result.fills.push(fill.clone());
                    self.trades.push(fill);

                    if bid_entry.0.remaining > 0 {
                        matched_bids.push(bid_entry);
                    }
                } else {
                    matched_bids.push(bid_entry);
                    break;
                }
            } else {
                break;
            }
        }

        // Put back unmatched bids
        for bid in matched_bids {
            self.bids.push(bid);
        }
    }

    /// Get total bid volume at top N levels
    pub fn bid_volume(&self, levels: usize) -> u64 {
        self.bids.iter()
            .take(levels)
            .map(|e| e.0.remaining)
            .sum()
    }

    /// Get total ask volume at top N levels
    pub fn ask_volume(&self, levels: usize) -> u64 {
        self.asks.iter()
            .take(levels)
            .map(|e| e.0.remaining)
            .sum()
    }

    /// Get trade history
    pub fn get_trades(&self) -> &[Fill] {
        &self.trades
    }

    /// Clear all orders
    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_limit_order_matching() {
        let mut book = OrderBook::new(100.0);

        // Add sell limit order
        let sell_order = Order::new(
            "seller".to_string(),
            Side::Sell,
            OrderType::Limit,
            101.0,
            10,
        );
        book.submit_order(sell_order);

        assert_eq!(book.best_ask(), Some(101.0));
        assert_eq!(book.best_bid(), None);

        // Add buy limit order that matches
        let buy_order = Order::new(
            "buyer".to_string(),
            Side::Buy,
            OrderType::Limit,
            101.0,
            5,
        );
        let result = book.submit_order(buy_order);

        assert_eq!(result.filled_quantity, 5);
        assert_eq!(result.average_price, 101.0);
    }

    #[test]
    fn test_market_order() {
        let mut book = OrderBook::new(100.0);

        // Add liquidity
        book.submit_order(Order::new(
            "mm".to_string(),
            Side::Sell,
            OrderType::Limit,
            100.5,
            100,
        ));

        // Market buy
        let result = book.submit_order(Order::new(
            "buyer".to_string(),
            Side::Buy,
            OrderType::Market,
            0.0, // Price ignored for market orders
            50,
        ));

        assert_eq!(result.filled_quantity, 50);
        assert_eq!(book.last_price, 100.5);
    }

    #[test]
    fn test_spread() {
        let mut book = OrderBook::new(100.0);

        book.submit_order(Order::new(
            "mm".to_string(),
            Side::Buy,
            OrderType::Limit,
            99.5,
            100,
        ));

        book.submit_order(Order::new(
            "mm".to_string(),
            Side::Sell,
            OrderType::Limit,
            100.5,
            100,
        ));

        assert_eq!(book.spread(), Some(1.0));
        assert_eq!(book.mid_price(), Some(100.0));
    }
}
