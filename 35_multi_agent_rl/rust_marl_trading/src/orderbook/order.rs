//! Order types and definitions

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::agents::AgentId;

/// Unique identifier for an order
pub type OrderId = Uuid;

/// Order side (buy or sell)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl OrderSide {
    /// Get the opposite side
    pub fn opposite(&self) -> Self {
        match self {
            OrderSide::Buy => OrderSide::Sell,
            OrderSide::Sell => OrderSide::Buy,
        }
    }

    /// Returns 1 for buy, -1 for sell
    pub fn sign(&self) -> f64 {
        match self {
            OrderSide::Buy => 1.0,
            OrderSide::Sell => -1.0,
        }
    }
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderType {
    /// Market order - executes immediately at best available price
    Market,
    /// Limit order - executes only at specified price or better
    Limit,
    /// Cancel order - cancels an existing order
    Cancel,
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderStatus {
    /// Order is pending in the order book
    Pending,
    /// Order has been partially filled
    PartiallyFilled,
    /// Order has been completely filled
    Filled,
    /// Order has been cancelled
    Cancelled,
    /// Order was rejected
    Rejected,
}

/// Represents an order in the order book
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Unique order ID
    pub id: OrderId,
    /// Agent who placed the order
    pub agent_id: AgentId,
    /// Order side (buy or sell)
    pub side: OrderSide,
    /// Order type (market or limit)
    pub order_type: OrderType,
    /// Price (for limit orders, None for market orders)
    pub price: Option<f64>,
    /// Original quantity
    pub quantity: f64,
    /// Remaining quantity
    pub remaining: f64,
    /// Timestamp when order was placed
    pub timestamp: DateTime<Utc>,
    /// Order status
    pub status: OrderStatus,
}

impl Order {
    /// Create a new order
    pub fn new(
        agent_id: AgentId,
        side: OrderSide,
        order_type: OrderType,
        price: Option<f64>,
        quantity: f64,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            agent_id,
            side,
            order_type,
            price,
            quantity,
            remaining: quantity,
            timestamp: Utc::now(),
            status: OrderStatus::Pending,
        }
    }

    /// Create a market buy order
    pub fn market_buy(agent_id: AgentId, quantity: f64) -> Self {
        Self::new(agent_id, OrderSide::Buy, OrderType::Market, None, quantity)
    }

    /// Create a market sell order
    pub fn market_sell(agent_id: AgentId, quantity: f64) -> Self {
        Self::new(agent_id, OrderSide::Sell, OrderType::Market, None, quantity)
    }

    /// Create a limit buy order
    pub fn limit_buy(agent_id: AgentId, price: f64, quantity: f64) -> Self {
        Self::new(
            agent_id,
            OrderSide::Buy,
            OrderType::Limit,
            Some(price),
            quantity,
        )
    }

    /// Create a limit sell order
    pub fn limit_sell(agent_id: AgentId, price: f64, quantity: f64) -> Self {
        Self::new(
            agent_id,
            OrderSide::Sell,
            OrderType::Limit,
            Some(price),
            quantity,
        )
    }

    /// Create a cancel order
    pub fn cancel(agent_id: AgentId, order_to_cancel: OrderId) -> Self {
        let mut order = Self::new(agent_id, OrderSide::Buy, OrderType::Cancel, None, 0.0);
        order.id = order_to_cancel;
        order
    }

    /// Check if order is completely filled
    pub fn is_filled(&self) -> bool {
        self.remaining <= 0.0 || self.status == OrderStatus::Filled
    }

    /// Check if order is active (pending or partially filled)
    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Pending | OrderStatus::PartiallyFilled
        )
    }

    /// Fill a portion of the order
    pub fn fill(&mut self, quantity: f64) {
        self.remaining = (self.remaining - quantity).max(0.0);
        if self.remaining <= 0.0 {
            self.status = OrderStatus::Filled;
        } else {
            self.status = OrderStatus::PartiallyFilled;
        }
    }

    /// Get filled quantity
    pub fn filled_quantity(&self) -> f64 {
        self.quantity - self.remaining
    }

    /// Get fill percentage
    pub fn fill_percentage(&self) -> f64 {
        if self.quantity > 0.0 {
            self.filled_quantity() / self.quantity
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_creation() {
        let order = Order::market_buy(0, 10.0);
        assert_eq!(order.side, OrderSide::Buy);
        assert_eq!(order.order_type, OrderType::Market);
        assert_eq!(order.quantity, 10.0);
        assert_eq!(order.remaining, 10.0);
        assert!(order.price.is_none());
    }

    #[test]
    fn test_limit_order() {
        let order = Order::limit_sell(1, 50000.0, 5.0);
        assert_eq!(order.side, OrderSide::Sell);
        assert_eq!(order.order_type, OrderType::Limit);
        assert_eq!(order.price, Some(50000.0));
    }

    #[test]
    fn test_order_fill() {
        let mut order = Order::market_buy(0, 10.0);
        order.fill(3.0);
        assert_eq!(order.remaining, 7.0);
        assert_eq!(order.status, OrderStatus::PartiallyFilled);

        order.fill(7.0);
        assert_eq!(order.remaining, 0.0);
        assert_eq!(order.status, OrderStatus::Filled);
        assert!(order.is_filled());
    }

    #[test]
    fn test_order_side() {
        assert_eq!(OrderSide::Buy.opposite(), OrderSide::Sell);
        assert_eq!(OrderSide::Sell.opposite(), OrderSide::Buy);
        assert_eq!(OrderSide::Buy.sign(), 1.0);
        assert_eq!(OrderSide::Sell.sign(), -1.0);
    }
}
