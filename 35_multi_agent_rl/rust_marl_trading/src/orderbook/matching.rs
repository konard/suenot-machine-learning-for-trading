//! Order matching engine

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{Order, OrderId, OrderSide};
use crate::agents::AgentId;

/// Unique identifier for a trade
pub type TradeId = Uuid;

/// Represents an executed trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Unique trade ID
    pub id: TradeId,
    /// Buyer agent ID
    pub buyer_id: AgentId,
    /// Seller agent ID
    pub seller_id: AgentId,
    /// Buy order ID
    pub buy_order_id: OrderId,
    /// Sell order ID
    pub sell_order_id: OrderId,
    /// Execution price
    pub price: f64,
    /// Executed quantity
    pub quantity: f64,
    /// Trade timestamp
    pub timestamp: DateTime<Utc>,
    /// Was the aggressor (taker) the buyer?
    pub buyer_is_taker: bool,
}

impl Trade {
    /// Create a new trade
    pub fn new(
        buyer_id: AgentId,
        seller_id: AgentId,
        buy_order_id: OrderId,
        sell_order_id: OrderId,
        price: f64,
        quantity: f64,
        buyer_is_taker: bool,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            buyer_id,
            seller_id,
            buy_order_id,
            sell_order_id,
            price,
            quantity,
            timestamp: Utc::now(),
            buyer_is_taker,
        }
    }

    /// Get trade value (price * quantity)
    pub fn value(&self) -> f64 {
        self.price * self.quantity
    }

    /// Get the taker agent ID
    pub fn taker_id(&self) -> AgentId {
        if self.buyer_is_taker {
            self.buyer_id
        } else {
            self.seller_id
        }
    }

    /// Get the maker agent ID
    pub fn maker_id(&self) -> AgentId {
        if self.buyer_is_taker {
            self.seller_id
        } else {
            self.buyer_id
        }
    }
}

/// Result of matching a single order
#[derive(Debug, Clone, Default)]
pub struct MatchResult {
    /// Trades executed
    pub trades: Vec<Trade>,
    /// IDs of orders that were fully filled and should be removed
    pub filled_orders: Vec<OrderId>,
    /// IDs of orders that were cancelled
    pub cancelled_orders: Vec<OrderId>,
    /// Total volume traded
    pub volume: f64,
    /// Volume-weighted average price
    pub vwap: f64,
}

impl MatchResult {
    /// Create an empty match result
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a trade to the result
    pub fn add_trade(&mut self, trade: Trade) {
        self.volume += trade.quantity;

        // Update VWAP
        let total_value: f64 = self.trades.iter().map(|t| t.value()).sum::<f64>() + trade.value();
        self.vwap = total_value / (self.volume);

        self.trades.push(trade);
    }

    /// Merge another match result into this one
    pub fn merge(&mut self, other: MatchResult) {
        let new_volume = self.volume + other.volume;
        if new_volume > 0.0 {
            self.vwap = (self.vwap * self.volume + other.vwap * other.volume) / new_volume;
        }
        self.volume = new_volume;
        self.trades.extend(other.trades);
        self.filled_orders.extend(other.filled_orders);
        self.cancelled_orders.extend(other.cancelled_orders);
    }

    /// Check if any trades occurred
    pub fn has_trades(&self) -> bool {
        !self.trades.is_empty()
    }

    /// Get number of trades
    pub fn trade_count(&self) -> usize {
        self.trades.len()
    }
}

/// Price level in the order book
#[derive(Debug, Clone)]
pub struct PriceLevel {
    pub price: f64,
    pub orders: Vec<Order>,
}

impl PriceLevel {
    pub fn new(price: f64) -> Self {
        Self {
            price,
            orders: Vec::new(),
        }
    }

    /// Add an order to this price level
    pub fn add_order(&mut self, order: Order) {
        self.orders.push(order);
    }

    /// Get total quantity at this level
    pub fn total_quantity(&self) -> f64 {
        self.orders.iter().map(|o| o.remaining).sum()
    }

    /// Get number of orders at this level
    pub fn order_count(&self) -> usize {
        self.orders.len()
    }

    /// Check if level is empty
    pub fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }

    /// Remove filled orders
    pub fn remove_filled(&mut self) -> Vec<OrderId> {
        let filled: Vec<OrderId> = self.orders
            .iter()
            .filter(|o| o.is_filled())
            .map(|o| o.id)
            .collect();

        self.orders.retain(|o| !o.is_filled());
        filled
    }
}

/// Match an incoming order against a price level
pub fn match_against_level(
    incoming: &mut Order,
    level: &mut PriceLevel,
) -> MatchResult {
    let mut result = MatchResult::new();

    // Determine if we can match at this price
    let can_match = match incoming.side {
        OrderSide::Buy => {
            // Buy order matches with asks at or below the buy price
            incoming.price.map_or(true, |p| level.price <= p)
        }
        OrderSide::Sell => {
            // Sell order matches with bids at or above the sell price
            incoming.price.map_or(true, |p| level.price >= p)
        }
    };

    if !can_match {
        return result;
    }

    // Match against orders in FIFO order (price-time priority)
    for resting in level.orders.iter_mut() {
        if incoming.remaining <= 0.0 {
            break;
        }

        // Skip orders from the same agent (no self-trading)
        if incoming.agent_id == resting.agent_id {
            continue;
        }

        let match_qty = incoming.remaining.min(resting.remaining);
        let match_price = level.price; // Use the resting order's price

        // Determine buyer and seller
        let (buyer_id, seller_id, buy_order_id, sell_order_id, buyer_is_taker) = match incoming.side {
            OrderSide::Buy => (
                incoming.agent_id,
                resting.agent_id,
                incoming.id,
                resting.id,
                true,
            ),
            OrderSide::Sell => (
                resting.agent_id,
                incoming.agent_id,
                resting.id,
                incoming.id,
                false,
            ),
        };

        // Execute the trade
        let trade = Trade::new(
            buyer_id,
            seller_id,
            buy_order_id,
            sell_order_id,
            match_price,
            match_qty,
            buyer_is_taker,
        );

        // Update order quantities
        incoming.fill(match_qty);
        resting.fill(match_qty);

        // Track filled orders
        if resting.is_filled() {
            result.filled_orders.push(resting.id);
        }

        result.add_trade(trade);
    }

    // Track if incoming order is filled
    if incoming.is_filled() {
        result.filled_orders.push(incoming.id);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trade_creation() {
        let trade = Trade::new(0, 1, Uuid::new_v4(), Uuid::new_v4(), 50000.0, 1.0, true);

        assert_eq!(trade.value(), 50000.0);
        assert_eq!(trade.taker_id(), 0);
        assert_eq!(trade.maker_id(), 1);
    }

    #[test]
    fn test_match_result() {
        let mut result = MatchResult::new();

        let trade1 = Trade::new(0, 1, Uuid::new_v4(), Uuid::new_v4(), 100.0, 10.0, true);
        let trade2 = Trade::new(0, 2, Uuid::new_v4(), Uuid::new_v4(), 101.0, 10.0, true);

        result.add_trade(trade1);
        result.add_trade(trade2);

        assert_eq!(result.trade_count(), 2);
        assert_eq!(result.volume, 20.0);
        assert!((result.vwap - 100.5).abs() < 0.01);
    }

    #[test]
    fn test_price_level() {
        let mut level = PriceLevel::new(50000.0);

        let order1 = Order::limit_sell(0, 50000.0, 10.0);
        let order2 = Order::limit_sell(1, 50000.0, 5.0);

        level.add_order(order1);
        level.add_order(order2);

        assert_eq!(level.total_quantity(), 15.0);
        assert_eq!(level.order_count(), 2);
    }

    #[test]
    fn test_match_against_level() {
        let mut level = PriceLevel::new(50000.0);
        level.add_order(Order::limit_sell(0, 50000.0, 10.0));
        level.add_order(Order::limit_sell(1, 50000.0, 5.0));

        let mut buy_order = Order::market_buy(2, 12.0);
        let result = match_against_level(&mut buy_order, &mut level);

        assert_eq!(result.trade_count(), 2);
        assert_eq!(result.volume, 12.0);
        assert_eq!(buy_order.remaining, 0.0);
    }
}
