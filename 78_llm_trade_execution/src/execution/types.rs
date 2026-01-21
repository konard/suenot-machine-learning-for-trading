//! Core execution types and order structures.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique order identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrderId(pub String);

impl OrderId {
    /// Generate a new random order ID
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create from a string
    pub fn from_string(s: String) -> Self {
        Self(s)
    }
}

impl Default for OrderId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for OrderId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Order side (buy or sell)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    /// Buy order
    Buy,
    /// Sell order
    Sell,
}

impl Side {
    /// Get the opposite side
    pub fn opposite(&self) -> Self {
        match self {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        }
    }

    /// Get the sign for calculations (+1 for buy, -1 for sell)
    pub fn sign(&self) -> f64 {
        match self {
            Side::Buy => 1.0,
            Side::Sell => -1.0,
        }
    }
}

impl std::fmt::Display for Side {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Side::Buy => write!(f, "Buy"),
            Side::Sell => write!(f, "Sell"),
        }
    }
}

/// Parent order status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParentOrderStatus {
    /// Order created but not started
    Pending,
    /// Order is being executed
    Active,
    /// Order is paused (can be resumed)
    Paused,
    /// Order completed successfully
    Completed,
    /// Order was cancelled
    Cancelled,
    /// Order failed due to error
    Failed,
}

impl ParentOrderStatus {
    /// Check if the order is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            ParentOrderStatus::Completed
                | ParentOrderStatus::Cancelled
                | ParentOrderStatus::Failed
        )
    }

    /// Check if the order is active (can generate child orders)
    pub fn is_active(&self) -> bool {
        matches!(self, ParentOrderStatus::Active)
    }
}

/// Parent order - the high-level order to be executed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParentOrder {
    /// Unique order identifier
    pub id: OrderId,
    /// Trading symbol
    pub symbol: String,
    /// Order side
    pub side: Side,
    /// Total quantity to execute
    pub total_quantity: f64,
    /// Quantity already filled
    pub filled_quantity: f64,
    /// Execution time horizon in seconds
    pub time_horizon: u64,
    /// Order status
    pub status: ParentOrderStatus,
    /// Arrival price (price when order was received)
    pub arrival_price: Option<f64>,
    /// Average fill price
    pub average_price: Option<f64>,
    /// Maximum participation rate (fraction of volume)
    pub max_participation: f64,
    /// Urgency parameter (0.0 = passive, 1.0 = aggressive)
    pub urgency: f64,
    /// Order creation time
    pub created_at: DateTime<Utc>,
    /// Order start time (when execution began)
    pub started_at: Option<DateTime<Utc>>,
    /// Order completion time
    pub completed_at: Option<DateTime<Utc>>,
    /// Price limit (optional)
    pub limit_price: Option<f64>,
    /// Client order ID (for exchange reference)
    pub client_order_id: Option<String>,
    /// Additional metadata
    pub metadata: Option<serde_json::Value>,
}

impl ParentOrder {
    /// Create a new parent order
    pub fn new(symbol: String, side: Side, total_quantity: f64, time_horizon: u64) -> Self {
        Self {
            id: OrderId::new(),
            symbol,
            side,
            total_quantity,
            filled_quantity: 0.0,
            time_horizon,
            status: ParentOrderStatus::Pending,
            arrival_price: None,
            average_price: None,
            max_participation: 0.10, // 10% default
            urgency: 0.5,            // Medium urgency
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            limit_price: None,
            client_order_id: None,
            metadata: None,
        }
    }

    /// Set the urgency parameter
    pub fn with_urgency(mut self, urgency: f64) -> Self {
        self.urgency = urgency.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum participation rate
    pub fn with_max_participation(mut self, rate: f64) -> Self {
        self.max_participation = rate.clamp(0.01, 0.50);
        self
    }

    /// Set a price limit
    pub fn with_limit_price(mut self, price: f64) -> Self {
        self.limit_price = Some(price);
        self
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Get remaining quantity to fill
    pub fn remaining_quantity(&self) -> f64 {
        (self.total_quantity - self.filled_quantity).max(0.0)
    }

    /// Get fill rate (fraction filled)
    pub fn fill_rate(&self) -> f64 {
        if self.total_quantity > 0.0 {
            self.filled_quantity / self.total_quantity
        } else {
            0.0
        }
    }

    /// Get remaining time in seconds
    pub fn remaining_time(&self) -> u64 {
        if let Some(started) = self.started_at {
            let elapsed = (Utc::now() - started).num_seconds() as u64;
            self.time_horizon.saturating_sub(elapsed)
        } else {
            self.time_horizon
        }
    }

    /// Get elapsed time in seconds
    pub fn elapsed_time(&self) -> u64 {
        if let Some(started) = self.started_at {
            (Utc::now() - started).num_seconds().max(0) as u64
        } else {
            0
        }
    }

    /// Check if the order should complete based on time
    pub fn is_time_expired(&self) -> bool {
        self.remaining_time() == 0
    }

    /// Record a fill
    pub fn record_fill(&mut self, quantity: f64, price: f64) {
        let new_filled = self.filled_quantity + quantity;
        let new_value = self.average_price.unwrap_or(0.0) * self.filled_quantity + price * quantity;

        self.filled_quantity = new_filled;
        self.average_price = Some(new_value / new_filled);

        // Check if completed
        if self.remaining_quantity() <= 0.0 {
            self.status = ParentOrderStatus::Completed;
            self.completed_at = Some(Utc::now());
        }
    }

    /// Start execution
    pub fn start(&mut self, arrival_price: f64) {
        self.status = ParentOrderStatus::Active;
        self.started_at = Some(Utc::now());
        self.arrival_price = Some(arrival_price);
    }

    /// Pause execution
    pub fn pause(&mut self) {
        if self.status == ParentOrderStatus::Active {
            self.status = ParentOrderStatus::Paused;
        }
    }

    /// Resume execution
    pub fn resume(&mut self) {
        if self.status == ParentOrderStatus::Paused {
            self.status = ParentOrderStatus::Active;
        }
    }

    /// Cancel the order
    pub fn cancel(&mut self) {
        if !self.status.is_terminal() {
            self.status = ParentOrderStatus::Cancelled;
            self.completed_at = Some(Utc::now());
        }
    }

    /// Mark as failed
    pub fn fail(&mut self) {
        self.status = ParentOrderStatus::Failed;
        self.completed_at = Some(Utc::now());
    }
}

/// Child order status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChildOrderStatus {
    /// Order created, waiting to send
    Pending,
    /// Order sent to exchange
    Sent,
    /// Order accepted by exchange
    Open,
    /// Order partially filled
    PartiallyFilled,
    /// Order completely filled
    Filled,
    /// Order cancelled
    Cancelled,
    /// Order rejected by exchange
    Rejected,
    /// Order expired
    Expired,
}

impl ChildOrderStatus {
    /// Check if the order is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            ChildOrderStatus::Filled
                | ChildOrderStatus::Cancelled
                | ChildOrderStatus::Rejected
                | ChildOrderStatus::Expired
        )
    }

    /// Check if the order is working (can be filled)
    pub fn is_working(&self) -> bool {
        matches!(
            self,
            ChildOrderStatus::Sent
                | ChildOrderStatus::Open
                | ChildOrderStatus::PartiallyFilled
        )
    }
}

/// Child order - individual slice sent to exchange
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChildOrder {
    /// Unique order identifier
    pub id: OrderId,
    /// Parent order ID
    pub parent_id: OrderId,
    /// Trading symbol
    pub symbol: String,
    /// Order side
    pub side: Side,
    /// Order quantity
    pub quantity: f64,
    /// Filled quantity
    pub filled_quantity: f64,
    /// Limit price (None for market order)
    pub limit_price: Option<f64>,
    /// Average fill price
    pub fill_price: Option<f64>,
    /// Order status
    pub status: ChildOrderStatus,
    /// Exchange order ID
    pub exchange_order_id: Option<String>,
    /// Order creation time
    pub created_at: DateTime<Utc>,
    /// Last update time
    pub updated_at: DateTime<Utc>,
    /// Slice index (for TWAP/VWAP)
    pub slice_index: Option<u32>,
    /// Time-in-force
    pub time_in_force: TimeInForce,
}

/// Time-in-force for orders
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeInForce {
    /// Good till cancelled
    GTC,
    /// Immediate or cancel
    IOC,
    /// Fill or kill
    FOK,
    /// Post only (maker)
    PostOnly,
}

impl Default for TimeInForce {
    fn default() -> Self {
        Self::IOC
    }
}

impl ChildOrder {
    /// Create a new child order
    pub fn new(
        parent_id: OrderId,
        symbol: String,
        side: Side,
        quantity: f64,
        limit_price: Option<f64>,
    ) -> Self {
        Self {
            id: OrderId::new(),
            parent_id,
            symbol,
            side,
            quantity,
            filled_quantity: 0.0,
            limit_price,
            fill_price: None,
            status: ChildOrderStatus::Pending,
            exchange_order_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            slice_index: None,
            time_in_force: TimeInForce::default(),
        }
    }

    /// Create a market order
    pub fn market(parent_id: OrderId, symbol: String, side: Side, quantity: f64) -> Self {
        Self::new(parent_id, symbol, side, quantity, None)
    }

    /// Create a limit order
    pub fn limit(
        parent_id: OrderId,
        symbol: String,
        side: Side,
        quantity: f64,
        price: f64,
    ) -> Self {
        Self::new(parent_id, symbol, side, quantity, Some(price))
    }

    /// Set slice index
    pub fn with_slice_index(mut self, index: u32) -> Self {
        self.slice_index = Some(index);
        self
    }

    /// Set time-in-force
    pub fn with_time_in_force(mut self, tif: TimeInForce) -> Self {
        self.time_in_force = tif;
        self
    }

    /// Get remaining quantity
    pub fn remaining_quantity(&self) -> f64 {
        (self.quantity - self.filled_quantity).max(0.0)
    }

    /// Check if order is a market order
    pub fn is_market(&self) -> bool {
        self.limit_price.is_none()
    }

    /// Record a fill
    pub fn record_fill(&mut self, quantity: f64, price: f64) {
        let new_filled = self.filled_quantity + quantity;
        let new_value = self.fill_price.unwrap_or(0.0) * self.filled_quantity + price * quantity;

        self.filled_quantity = new_filled;
        self.fill_price = Some(new_value / new_filled);
        self.updated_at = Utc::now();

        // Update status
        if self.remaining_quantity() <= 0.0 {
            self.status = ChildOrderStatus::Filled;
        } else {
            self.status = ChildOrderStatus::PartiallyFilled;
        }
    }

    /// Mark as sent
    pub fn mark_sent(&mut self, exchange_order_id: Option<String>) {
        self.status = ChildOrderStatus::Sent;
        self.exchange_order_id = exchange_order_id;
        self.updated_at = Utc::now();
    }

    /// Mark as open
    pub fn mark_open(&mut self) {
        self.status = ChildOrderStatus::Open;
        self.updated_at = Utc::now();
    }

    /// Mark as cancelled
    pub fn mark_cancelled(&mut self) {
        self.status = ChildOrderStatus::Cancelled;
        self.updated_at = Utc::now();
    }

    /// Mark as rejected
    pub fn mark_rejected(&mut self) {
        self.status = ChildOrderStatus::Rejected;
        self.updated_at = Utc::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parent_order_creation() {
        let order = ParentOrder::new("BTCUSDT".to_string(), Side::Buy, 10.0, 3600);

        assert_eq!(order.symbol, "BTCUSDT");
        assert_eq!(order.side, Side::Buy);
        assert_eq!(order.total_quantity, 10.0);
        assert_eq!(order.filled_quantity, 0.0);
        assert_eq!(order.status, ParentOrderStatus::Pending);
    }

    #[test]
    fn test_parent_order_fill() {
        let mut order = ParentOrder::new("BTCUSDT".to_string(), Side::Buy, 10.0, 3600);
        order.start(50000.0);

        order.record_fill(5.0, 50100.0);
        assert_eq!(order.filled_quantity, 5.0);
        assert_eq!(order.average_price, Some(50100.0));
        assert_eq!(order.remaining_quantity(), 5.0);
        assert!(!order.status.is_terminal());

        order.record_fill(5.0, 50200.0);
        assert_eq!(order.filled_quantity, 10.0);
        assert_eq!(order.average_price, Some(50150.0));
        assert_eq!(order.status, ParentOrderStatus::Completed);
    }

    #[test]
    fn test_child_order_fill() {
        let parent_id = OrderId::new();
        let mut child = ChildOrder::market(parent_id, "BTCUSDT".to_string(), Side::Buy, 1.0);

        assert!(child.is_market());
        assert_eq!(child.remaining_quantity(), 1.0);

        child.record_fill(0.5, 50000.0);
        assert_eq!(child.status, ChildOrderStatus::PartiallyFilled);
        assert_eq!(child.fill_price, Some(50000.0));

        child.record_fill(0.5, 50100.0);
        assert_eq!(child.status, ChildOrderStatus::Filled);
        assert_eq!(child.fill_price, Some(50050.0));
    }

    #[test]
    fn test_side_operations() {
        assert_eq!(Side::Buy.opposite(), Side::Sell);
        assert_eq!(Side::Sell.opposite(), Side::Buy);
        assert_eq!(Side::Buy.sign(), 1.0);
        assert_eq!(Side::Sell.sign(), -1.0);
    }
}
