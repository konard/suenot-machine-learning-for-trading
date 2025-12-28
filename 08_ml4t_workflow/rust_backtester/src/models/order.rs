//! Order data model.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Order side (buy or sell).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl std::fmt::Display for OrderSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderSide::Buy => write!(f, "BUY"),
            OrderSide::Sell => write!(f, "SELL"),
        }
    }
}

/// Order type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    StopLoss,
    TakeProfit,
}

/// Order status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    Pending,
    Filled,
    PartiallyFilled,
    Cancelled,
    Rejected,
}

/// Represents a trading order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Unique order ID
    pub id: String,
    /// Symbol (e.g., "BTCUSDT")
    pub symbol: String,
    /// Order side
    pub side: OrderSide,
    /// Order type
    pub order_type: OrderType,
    /// Order quantity
    pub quantity: f64,
    /// Limit price (for limit orders)
    pub price: Option<f64>,
    /// Stop price (for stop orders)
    pub stop_price: Option<f64>,
    /// Order status
    pub status: OrderStatus,
    /// Filled quantity
    pub filled_quantity: f64,
    /// Average fill price
    pub avg_fill_price: f64,
    /// Order creation time
    pub created_at: DateTime<Utc>,
    /// Order update time
    pub updated_at: DateTime<Utc>,
}

impl Order {
    /// Create a new market order.
    pub fn market(id: String, symbol: String, side: OrderSide, quantity: f64) -> Self {
        let now = Utc::now();
        Self {
            id,
            symbol,
            side,
            order_type: OrderType::Market,
            quantity,
            price: None,
            stop_price: None,
            status: OrderStatus::Pending,
            filled_quantity: 0.0,
            avg_fill_price: 0.0,
            created_at: now,
            updated_at: now,
        }
    }

    /// Create a new limit order.
    pub fn limit(id: String, symbol: String, side: OrderSide, quantity: f64, price: f64) -> Self {
        let now = Utc::now();
        Self {
            id,
            symbol,
            side,
            order_type: OrderType::Limit,
            quantity,
            price: Some(price),
            stop_price: None,
            status: OrderStatus::Pending,
            filled_quantity: 0.0,
            avg_fill_price: 0.0,
            created_at: now,
            updated_at: now,
        }
    }

    /// Check if the order is completely filled.
    pub fn is_filled(&self) -> bool {
        self.status == OrderStatus::Filled
    }

    /// Check if the order is still active.
    pub fn is_active(&self) -> bool {
        matches!(self.status, OrderStatus::Pending | OrderStatus::PartiallyFilled)
    }

    /// Calculate the remaining quantity to fill.
    pub fn remaining_quantity(&self) -> f64 {
        self.quantity - self.filled_quantity
    }

    /// Fill the order at the given price.
    pub fn fill(&mut self, price: f64, quantity: f64) {
        let fill_qty = quantity.min(self.remaining_quantity());
        let total_value = self.avg_fill_price * self.filled_quantity + price * fill_qty;
        self.filled_quantity += fill_qty;
        self.avg_fill_price = total_value / self.filled_quantity;
        self.updated_at = Utc::now();

        if (self.filled_quantity - self.quantity).abs() < 1e-10 {
            self.status = OrderStatus::Filled;
        } else {
            self.status = OrderStatus::PartiallyFilled;
        }
    }
}
