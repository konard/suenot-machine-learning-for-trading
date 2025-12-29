//! Order book implementation

use std::collections::BTreeMap;

use super::matching::{match_against_level, MatchResult, PriceLevel};
use super::order::{Order, OrderId, OrderSide, OrderStatus, OrderType};
use crate::agents::AgentId;

/// Comparison wrapper for f64 to use in BTreeMap
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

/// Order book with price-time priority matching
#[derive(Debug, Clone)]
pub struct OrderBook {
    /// Buy orders (bids), sorted by price descending
    bids: BTreeMap<OrderedFloat, PriceLevel>,
    /// Sell orders (asks), sorted by price ascending
    asks: BTreeMap<OrderedFloat, PriceLevel>,
    /// Last traded price
    last_price: Option<f64>,
    /// Fundamental value (for informed traders)
    fundamental_value: f64,
    /// Total volume traded
    total_volume: f64,
    /// Order lookup by ID
    order_map: std::collections::HashMap<OrderId, (OrderSide, f64)>,
}

impl Default for OrderBook {
    fn default() -> Self {
        Self::new(100.0)
    }
}

impl OrderBook {
    /// Create a new order book
    pub fn new(initial_price: f64) -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_price: Some(initial_price),
            fundamental_value: initial_price,
            total_volume: 0.0,
            order_map: std::collections::HashMap::new(),
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
    pub fn mid_price(&self) -> f64 {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => (bid + ask) / 2.0,
            (Some(bid), None) => bid,
            (None, Some(ask)) => ask,
            (None, None) => self.last_price.unwrap_or(self.fundamental_value),
        }
    }

    /// Get spread
    pub fn spread(&self) -> f64 {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => ask - bid,
            _ => 0.0,
        }
    }

    /// Get spread as percentage
    pub fn spread_pct(&self) -> f64 {
        let mid = self.mid_price();
        if mid > 0.0 {
            self.spread() / mid * 100.0
        } else {
            0.0
        }
    }

    /// Get last traded price
    pub fn last_price(&self) -> f64 {
        self.last_price.unwrap_or(self.mid_price())
    }

    /// Set fundamental value
    pub fn set_fundamental_value(&mut self, value: f64) {
        self.fundamental_value = value;
    }

    /// Get fundamental value
    pub fn fundamental_value(&self) -> f64 {
        self.fundamental_value
    }

    /// Get total bid depth
    pub fn bid_depth(&self) -> f64 {
        self.bids.values().map(|level| level.total_quantity()).sum()
    }

    /// Get total ask depth
    pub fn ask_depth(&self) -> f64 {
        self.asks.values().map(|level| level.total_quantity()).sum()
    }

    /// Get order book imbalance (-1 to 1)
    pub fn imbalance(&self) -> f64 {
        let bid_depth = self.bid_depth();
        let ask_depth = self.ask_depth();
        let total = bid_depth + ask_depth;

        if total > 0.0 {
            (bid_depth - ask_depth) / total
        } else {
            0.0
        }
    }

    /// Get bid depth at a specific number of levels
    pub fn bid_depth_levels(&self, n_levels: usize) -> Vec<(f64, f64)> {
        self.bids
            .iter()
            .rev()
            .take(n_levels)
            .map(|(price, level)| (price.0, level.total_quantity()))
            .collect()
    }

    /// Get ask depth at a specific number of levels
    pub fn ask_depth_levels(&self, n_levels: usize) -> Vec<(f64, f64)> {
        self.asks
            .iter()
            .take(n_levels)
            .map(|(price, level)| (price.0, level.total_quantity()))
            .collect()
    }

    /// Get total volume traded
    pub fn total_volume(&self) -> f64 {
        self.total_volume
    }

    /// Submit an order and match it
    pub fn submit_order(&mut self, mut order: Order) -> MatchResult {
        let mut result = MatchResult::new();

        match order.order_type {
            OrderType::Cancel => {
                return self.cancel_order(order.id, order.agent_id);
            }
            OrderType::Market | OrderType::Limit => {
                // Try to match the order
                result = self.match_order(&mut order);

                // If limit order has remaining quantity, add to book
                if order.order_type == OrderType::Limit && order.remaining > 0.0 && order.is_active() {
                    self.add_to_book(order);
                }
            }
        }

        // Update last price if there were trades
        if let Some(trade) = result.trades.last() {
            self.last_price = Some(trade.price);
            self.total_volume += result.volume;
        }

        // Clean up filled orders
        self.cleanup_filled_orders(&result.filled_orders);

        result
    }

    /// Submit multiple orders simultaneously (batch processing)
    pub fn submit_orders(&mut self, orders: Vec<Order>) -> MatchResult {
        let mut combined_result = MatchResult::new();

        // Sort orders by timestamp for fair processing
        let mut orders = orders;
        orders.sort_by_key(|o| o.timestamp);

        for order in orders {
            let result = self.submit_order(order);
            combined_result.merge(result);
        }

        combined_result
    }

    /// Match an order against the opposite side of the book
    fn match_order(&mut self, order: &mut Order) -> MatchResult {
        let mut result = MatchResult::new();

        let opposite_book = match order.side {
            OrderSide::Buy => &mut self.asks,
            OrderSide::Sell => &mut self.bids,
        };

        // Get prices to match against
        let prices: Vec<OrderedFloat> = match order.side {
            OrderSide::Buy => opposite_book.keys().cloned().collect(),
            OrderSide::Sell => opposite_book.keys().rev().cloned().collect(),
        };

        for price in prices {
            if order.remaining <= 0.0 {
                break;
            }

            // Check if price is acceptable
            let acceptable = match order.side {
                OrderSide::Buy => order.price.map_or(true, |p| price.0 <= p),
                OrderSide::Sell => order.price.map_or(true, |p| price.0 >= p),
            };

            if !acceptable {
                break; // No more acceptable prices
            }

            if let Some(level) = opposite_book.get_mut(&price) {
                let level_result = match_against_level(order, level);
                result.merge(level_result);
            }
        }

        result
    }

    /// Add a limit order to the book
    fn add_to_book(&mut self, order: Order) {
        let price = match order.price {
            Some(p) => p,
            None => return, // Market orders can't be added to book
        };

        let key = OrderedFloat(price);

        // Track order location
        self.order_map.insert(order.id, (order.side, price));

        let book = match order.side {
            OrderSide::Buy => &mut self.bids,
            OrderSide::Sell => &mut self.asks,
        };

        book.entry(key)
            .or_insert_with(|| PriceLevel::new(price))
            .add_order(order);
    }

    /// Cancel an order
    fn cancel_order(&mut self, order_id: OrderId, agent_id: AgentId) -> MatchResult {
        let mut result = MatchResult::new();

        if let Some((side, price)) = self.order_map.remove(&order_id) {
            let key = OrderedFloat(price);

            let book = match side {
                OrderSide::Buy => &mut self.bids,
                OrderSide::Sell => &mut self.asks,
            };

            if let Some(level) = book.get_mut(&key) {
                // Find and remove the order (only if it belongs to the agent)
                level.orders.retain(|o| !(o.id == order_id && o.agent_id == agent_id));
                result.cancelled_orders.push(order_id);

                // Remove empty levels
                if level.is_empty() {
                    book.remove(&key);
                }
            }
        }

        result
    }

    /// Clean up filled orders from the book
    fn cleanup_filled_orders(&mut self, filled: &[OrderId]) {
        for order_id in filled {
            if let Some((side, price)) = self.order_map.remove(order_id) {
                let key = OrderedFloat(price);

                let book = match side {
                    OrderSide::Buy => &mut self.bids,
                    OrderSide::Sell => &mut self.asks,
                };

                if let Some(level) = book.get_mut(&key) {
                    level.orders.retain(|o| o.id != *order_id);

                    if level.is_empty() {
                        book.remove(&key);
                    }
                }
            }
        }
    }

    /// Cancel all orders for an agent
    pub fn cancel_all_for_agent(&mut self, agent_id: AgentId) -> MatchResult {
        let mut result = MatchResult::new();

        // Find all orders for this agent
        let orders_to_cancel: Vec<OrderId> = self.order_map
            .iter()
            .filter_map(|(order_id, _)| {
                // We need to check if the order belongs to this agent
                // This requires looking in the book
                for level in self.bids.values() {
                    if let Some(order) = level.orders.iter().find(|o| o.id == *order_id) {
                        if order.agent_id == agent_id {
                            return Some(*order_id);
                        }
                    }
                }
                for level in self.asks.values() {
                    if let Some(order) = level.orders.iter().find(|o| o.id == *order_id) {
                        if order.agent_id == agent_id {
                            return Some(*order_id);
                        }
                    }
                }
                None
            })
            .collect();

        for order_id in orders_to_cancel {
            let cancel_result = self.cancel_order(order_id, agent_id);
            result.merge(cancel_result);
        }

        result
    }

    /// Get snapshot of current book state
    pub fn snapshot(&self, depth: usize) -> OrderBookSnapshot {
        OrderBookSnapshot {
            mid_price: self.mid_price(),
            spread: self.spread(),
            bids: self.bid_depth_levels(depth),
            asks: self.ask_depth_levels(depth),
            imbalance: self.imbalance(),
            last_price: self.last_price(),
        }
    }

    /// Reset the order book
    pub fn reset(&mut self, initial_price: f64) {
        self.bids.clear();
        self.asks.clear();
        self.order_map.clear();
        self.last_price = Some(initial_price);
        self.fundamental_value = initial_price;
        self.total_volume = 0.0;
    }
}

/// Snapshot of order book state
#[derive(Debug, Clone)]
pub struct OrderBookSnapshot {
    pub mid_price: f64,
    pub spread: f64,
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
    pub imbalance: f64,
    pub last_price: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_book_creation() {
        let book = OrderBook::new(50000.0);
        assert_eq!(book.mid_price(), 50000.0);
        assert_eq!(book.spread(), 0.0);
    }

    #[test]
    fn test_limit_order_submission() {
        let mut book = OrderBook::new(50000.0);

        // Add bid
        let bid = Order::limit_buy(0, 49990.0, 10.0);
        book.submit_order(bid);

        // Add ask
        let ask = Order::limit_sell(1, 50010.0, 10.0);
        book.submit_order(ask);

        assert_eq!(book.best_bid(), Some(49990.0));
        assert_eq!(book.best_ask(), Some(50010.0));
        assert_eq!(book.spread(), 20.0);
    }

    #[test]
    fn test_market_order_matching() {
        let mut book = OrderBook::new(50000.0);

        // Add asks
        book.submit_order(Order::limit_sell(0, 50010.0, 10.0));
        book.submit_order(Order::limit_sell(1, 50020.0, 10.0));

        // Submit market buy
        let market_buy = Order::market_buy(2, 15.0);
        let result = book.submit_order(market_buy);

        assert_eq!(result.trade_count(), 2);
        assert_eq!(result.volume, 15.0);

        // Check remaining ask quantity
        assert_eq!(book.ask_depth(), 5.0);
    }

    #[test]
    fn test_no_self_trading() {
        let mut book = OrderBook::new(50000.0);

        // Add ask from agent 0
        book.submit_order(Order::limit_sell(0, 50010.0, 10.0));

        // Submit buy from same agent
        let buy = Order::market_buy(0, 5.0);
        let result = book.submit_order(buy);

        // Should not match
        assert_eq!(result.trade_count(), 0);
    }

    #[test]
    fn test_order_cancellation() {
        let mut book = OrderBook::new(50000.0);

        let order = Order::limit_sell(0, 50010.0, 10.0);
        let order_id = order.id;
        book.submit_order(order);

        assert_eq!(book.ask_depth(), 10.0);

        // Cancel the order
        let cancel = Order::cancel(0, order_id);
        let result = book.submit_order(cancel);

        assert_eq!(result.cancelled_orders.len(), 1);
        assert_eq!(book.ask_depth(), 0.0);
    }

    #[test]
    fn test_imbalance() {
        let mut book = OrderBook::new(50000.0);

        book.submit_order(Order::limit_buy(0, 49990.0, 100.0));
        book.submit_order(Order::limit_sell(1, 50010.0, 50.0));

        // More bids than asks -> positive imbalance
        let imbalance = book.imbalance();
        assert!(imbalance > 0.0);
        assert!((imbalance - 0.333).abs() < 0.01); // (100-50)/(100+50) ≈ 0.333
    }
}
