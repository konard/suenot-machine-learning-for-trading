//! Order execution and position management

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Order side
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

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    StopMarket,
    TakeProfit,
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

/// Trading order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Order ID
    pub id: String,
    /// Symbol
    pub symbol: String,
    /// Order side
    pub side: OrderSide,
    /// Order type
    pub order_type: OrderType,
    /// Order size
    pub size: f64,
    /// Limit price (for limit orders)
    pub price: Option<f64>,
    /// Stop price (for stop orders)
    pub stop_price: Option<f64>,
    /// Order status
    pub status: OrderStatus,
    /// Filled size
    pub filled_size: f64,
    /// Average fill price
    pub avg_fill_price: f64,
    /// Created timestamp
    pub created_at: u64,
    /// Updated timestamp
    pub updated_at: u64,
}

impl Order {
    /// Create a new order
    pub fn new(
        symbol: impl Into<String>,
        side: OrderSide,
        order_type: OrderType,
        size: f64,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            id: uuid::Uuid::new_v4().to_string(),
            symbol: symbol.into(),
            side,
            order_type,
            size,
            price: None,
            stop_price: None,
            status: OrderStatus::New,
            filled_size: 0.0,
            avg_fill_price: 0.0,
            created_at: now,
            updated_at: now,
        }
    }

    /// Create a market order
    pub fn market(symbol: impl Into<String>, side: OrderSide, size: f64, _price: f64) -> Self {
        Self::new(symbol, side, OrderType::Market, size)
    }

    /// Create a limit order
    pub fn limit(symbol: impl Into<String>, side: OrderSide, size: f64, price: f64) -> Self {
        let mut order = Self::new(symbol, side, OrderType::Limit, size);
        order.price = Some(price);
        order
    }

    /// Create a stop market order
    pub fn stop_market(
        symbol: impl Into<String>,
        side: OrderSide,
        size: f64,
        stop_price: f64,
    ) -> Self {
        let mut order = Self::new(symbol, side, OrderType::StopMarket, size);
        order.stop_price = Some(stop_price);
        order
    }

    /// Check if order is filled
    pub fn is_filled(&self) -> bool {
        self.status == OrderStatus::Filled
    }

    /// Check if order is active
    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::New | OrderStatus::PartiallyFilled
        )
    }

    /// Remaining size
    pub fn remaining_size(&self) -> f64 {
        self.size - self.filled_size
    }

    /// Fill the order (simulate)
    pub fn fill(&mut self, fill_size: f64, fill_price: f64) {
        let total_value = self.avg_fill_price * self.filled_size + fill_price * fill_size;
        self.filled_size += fill_size;
        self.avg_fill_price = total_value / self.filled_size;

        if self.filled_size >= self.size {
            self.status = OrderStatus::Filled;
        } else {
            self.status = OrderStatus::PartiallyFilled;
        }

        self.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
    }

    /// Cancel the order
    pub fn cancel(&mut self) {
        self.status = OrderStatus::Cancelled;
        self.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
    }
}

// Simple UUID implementation for no_std compatibility
mod uuid {
    use rand::Rng;

    pub struct Uuid([u8; 16]);

    impl Uuid {
        pub fn new_v4() -> Self {
            let mut bytes = [0u8; 16];
            let mut rng = rand::thread_rng();
            rng.fill(&mut bytes);
            // Set version 4
            bytes[6] = (bytes[6] & 0x0f) | 0x40;
            // Set variant
            bytes[8] = (bytes[8] & 0x3f) | 0x80;
            Uuid(bytes)
        }

        pub fn to_string(&self) -> String {
            format!(
                "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                self.0[0], self.0[1], self.0[2], self.0[3],
                self.0[4], self.0[5],
                self.0[6], self.0[7],
                self.0[8], self.0[9],
                self.0[10], self.0[11], self.0[12], self.0[13], self.0[14], self.0[15]
            )
        }
    }
}

/// A trading position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Symbol
    pub symbol: String,
    /// Position side
    pub side: OrderSide,
    /// Position size
    pub size: f64,
    /// Entry price
    pub entry_price: f64,
    /// Current unrealized PnL
    pub unrealized_pnl: f64,
    /// Realized PnL
    pub realized_pnl: f64,
    /// Entry timestamp
    pub entry_time: u64,
    /// Last update timestamp
    pub updated_at: u64,
}

impl Position {
    /// Create a new position
    pub fn new(
        symbol: impl Into<String>,
        side: OrderSide,
        size: f64,
        entry_price: f64,
        entry_time: u64,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            side,
            size,
            entry_price,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            entry_time,
            updated_at: entry_time,
        }
    }

    /// Update position with current price
    pub fn update_pnl(&mut self, current_price: f64) {
        self.unrealized_pnl = match self.side {
            OrderSide::Buy => (current_price - self.entry_price) * self.size,
            OrderSide::Sell => (self.entry_price - current_price) * self.size,
        };

        self.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
    }

    /// Get return percentage
    pub fn return_pct(&self) -> f64 {
        let notional = self.entry_price * self.size;
        if notional > 0.0 {
            self.unrealized_pnl / notional
        } else {
            0.0
        }
    }

    /// Get position notional value
    pub fn notional(&self) -> f64 {
        self.entry_price * self.size
    }

    /// Get position holding time in seconds
    pub fn holding_time(&self) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        (now - self.entry_time) / 1000
    }
}

/// Position manager
#[derive(Debug)]
pub struct PositionManager {
    /// Current positions
    positions: HashMap<String, Position>,
    /// Maximum positions
    max_positions: usize,
    /// Closed positions history
    closed_positions: Vec<Position>,
    /// Total realized PnL
    pub total_realized_pnl: f64,
}

impl PositionManager {
    /// Create a new position manager
    pub fn new(max_positions: usize) -> Self {
        Self {
            positions: HashMap::new(),
            max_positions,
            closed_positions: Vec::new(),
            total_realized_pnl: 0.0,
        }
    }

    /// Open a new position
    pub fn open_position(
        &mut self,
        symbol: &str,
        side: OrderSide,
        size: f64,
        entry_price: f64,
        entry_time: u64,
    ) -> bool {
        if self.positions.len() >= self.max_positions {
            return false;
        }

        if self.positions.contains_key(symbol) {
            return false;
        }

        let position = Position::new(symbol, side, size, entry_price, entry_time);
        self.positions.insert(symbol.to_string(), position);
        true
    }

    /// Close a position
    pub fn close_position(&mut self, symbol: &str) -> Option<f64> {
        if let Some(mut position) = self.positions.remove(symbol) {
            position.realized_pnl = position.unrealized_pnl;
            self.total_realized_pnl += position.realized_pnl;
            let pnl = position.realized_pnl;
            self.closed_positions.push(position);
            Some(pnl)
        } else {
            None
        }
    }

    /// Get a position
    pub fn get_position(&self, symbol: &str) -> Option<&Position> {
        self.positions.get(symbol)
    }

    /// Get mutable position
    pub fn get_position_mut(&mut self, symbol: &str) -> Option<&mut Position> {
        self.positions.get_mut(symbol)
    }

    /// Update all positions with current prices
    pub fn update_prices(&mut self, prices: &HashMap<String, f64>) {
        for (symbol, position) in &mut self.positions {
            if let Some(&price) = prices.get(symbol) {
                position.update_pnl(price);
            }
        }
    }

    /// Get all open positions
    pub fn all_positions(&self) -> Vec<&Position> {
        self.positions.values().collect()
    }

    /// Get number of open positions
    pub fn position_count(&self) -> usize {
        self.positions.len()
    }

    /// Get total unrealized PnL
    pub fn total_unrealized_pnl(&self) -> f64 {
        self.positions.values().map(|p| p.unrealized_pnl).sum()
    }

    /// Get total notional
    pub fn total_notional(&self) -> f64 {
        self.positions.values().map(|p| p.notional()).sum()
    }

    /// Check if can open new position
    pub fn can_open_position(&self) -> bool {
        self.positions.len() < self.max_positions
    }

    /// Get position statistics
    pub fn stats(&self) -> PositionStats {
        let positions: Vec<_> = self.positions.values().collect();
        let total = positions.len();

        let long_count = positions.iter().filter(|p| p.side == OrderSide::Buy).count();
        let short_count = total - long_count;

        let winning = positions.iter().filter(|p| p.unrealized_pnl > 0.0).count();
        let losing = positions.iter().filter(|p| p.unrealized_pnl < 0.0).count();

        let avg_return = if total > 0 {
            positions.iter().map(|p| p.return_pct()).sum::<f64>() / total as f64
        } else {
            0.0
        };

        PositionStats {
            total_positions: total,
            long_positions: long_count,
            short_positions: short_count,
            winning_positions: winning,
            losing_positions: losing,
            total_unrealized_pnl: self.total_unrealized_pnl(),
            total_realized_pnl: self.total_realized_pnl,
            avg_return_pct: avg_return,
        }
    }

    /// Get closed positions history
    pub fn closed_history(&self) -> &[Position] {
        &self.closed_positions
    }
}

/// Position statistics
#[derive(Debug, Clone)]
pub struct PositionStats {
    pub total_positions: usize,
    pub long_positions: usize,
    pub short_positions: usize,
    pub winning_positions: usize,
    pub losing_positions: usize,
    pub total_unrealized_pnl: f64,
    pub total_realized_pnl: f64,
    pub avg_return_pct: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_creation() {
        let order = Order::market("BTCUSDT", OrderSide::Buy, 1.0, 50000.0);
        assert_eq!(order.side, OrderSide::Buy);
        assert_eq!(order.order_type, OrderType::Market);
        assert_eq!(order.size, 1.0);
    }

    #[test]
    fn test_order_fill() {
        let mut order = Order::market("BTCUSDT", OrderSide::Buy, 1.0, 50000.0);
        order.fill(0.5, 50100.0);

        assert_eq!(order.status, OrderStatus::PartiallyFilled);
        assert_eq!(order.filled_size, 0.5);

        order.fill(0.5, 50200.0);
        assert_eq!(order.status, OrderStatus::Filled);
        assert!(order.avg_fill_price > 50000.0);
    }

    #[test]
    fn test_position() {
        let mut position = Position::new("BTCUSDT", OrderSide::Buy, 1.0, 50000.0, 1000);

        // Price goes up
        position.update_pnl(51000.0);
        assert_eq!(position.unrealized_pnl, 1000.0);
        assert_eq!(position.return_pct(), 0.02); // 2% return
    }

    #[test]
    fn test_position_manager() {
        let mut manager = PositionManager::new(5);

        // Open position
        assert!(manager.open_position("BTCUSDT", OrderSide::Buy, 1.0, 50000.0, 1000));
        assert_eq!(manager.position_count(), 1);

        // Can't open duplicate
        assert!(!manager.open_position("BTCUSDT", OrderSide::Buy, 1.0, 50000.0, 1000));

        // Update and close
        let mut prices = HashMap::new();
        prices.insert("BTCUSDT".to_string(), 51000.0);
        manager.update_prices(&prices);

        let pnl = manager.close_position("BTCUSDT").unwrap();
        assert_eq!(pnl, 1000.0);
        assert_eq!(manager.position_count(), 0);
        assert_eq!(manager.total_realized_pnl, 1000.0);
    }

    #[test]
    fn test_max_positions() {
        let mut manager = PositionManager::new(2);

        assert!(manager.open_position("BTC", OrderSide::Buy, 1.0, 50000.0, 1000));
        assert!(manager.open_position("ETH", OrderSide::Buy, 1.0, 3000.0, 1000));
        assert!(!manager.open_position("SOL", OrderSide::Buy, 1.0, 100.0, 1000));

        assert!(!manager.can_open_position());
    }
}
