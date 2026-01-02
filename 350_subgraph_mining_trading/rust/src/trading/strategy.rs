//! Trading strategy implementation.

use crate::trading::{Signal, SignalType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Position in a symbol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Symbol
    pub symbol: String,
    /// Quantity (positive = long, negative = short)
    pub quantity: f64,
    /// Entry price
    pub entry_price: f64,
    /// Current price
    pub current_price: f64,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
    /// Entry timestamp
    pub entry_time: i64,
}

impl Position {
    /// Create a new position
    pub fn new(symbol: impl Into<String>, quantity: f64, entry_price: f64) -> Self {
        Self {
            symbol: symbol.into(),
            quantity,
            entry_price,
            current_price: entry_price,
            unrealized_pnl: 0.0,
            entry_time: chrono::Utc::now().timestamp(),
        }
    }

    /// Update position with new price
    pub fn update_price(&mut self, price: f64) {
        self.current_price = price;
        self.unrealized_pnl = (price - self.entry_price) * self.quantity;
    }

    /// Check if position is long
    pub fn is_long(&self) -> bool {
        self.quantity > 0.0
    }

    /// Check if position is short
    pub fn is_short(&self) -> bool {
        self.quantity < 0.0
    }

    /// Get position value
    pub fn value(&self) -> f64 {
        self.quantity.abs() * self.current_price
    }

    /// Get return percentage
    pub fn return_pct(&self) -> f64 {
        if self.entry_price == 0.0 {
            return 0.0;
        }
        (self.current_price - self.entry_price) / self.entry_price * self.quantity.signum()
    }
}

/// Strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Maximum position size (fraction of capital)
    pub max_position_size: f64,
    /// Minimum signal strength to act on
    pub min_signal_strength: f64,
    /// Stop loss percentage
    pub stop_loss_pct: f64,
    /// Take profit percentage
    pub take_profit_pct: f64,
    /// Maximum number of positions
    pub max_positions: usize,
    /// Whether to allow shorting
    pub allow_short: bool,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            max_position_size: 0.1, // 10% per position
            min_signal_strength: 0.6,
            stop_loss_pct: 0.05, // 5%
            take_profit_pct: 0.10, // 10%
            max_positions: 5,
            allow_short: false,
        }
    }
}

/// Trading strategy based on subgraph patterns
#[derive(Debug, Clone)]
pub struct Strategy {
    /// Configuration
    config: StrategyConfig,
    /// Current positions
    positions: HashMap<String, Position>,
    /// Available capital
    available_capital: f64,
    /// Total capital (positions + available)
    total_capital: f64,
    /// Trade history
    trade_history: Vec<TradeRecord>,
}

/// Record of a trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub symbol: String,
    pub side: TradeSide,
    pub quantity: f64,
    pub price: f64,
    pub timestamp: i64,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

impl Strategy {
    /// Create a new strategy
    pub fn new(config: StrategyConfig) -> Self {
        let capital = config.initial_capital;
        Self {
            config,
            positions: HashMap::new(),
            available_capital: capital,
            total_capital: capital,
            trade_history: Vec::new(),
        }
    }

    /// Process signals and generate orders
    pub fn process_signals(
        &mut self,
        signals: &[Signal],
        prices: &HashMap<String, f64>,
    ) -> Vec<Order> {
        let mut orders = Vec::new();

        // Update existing positions with new prices
        self.update_positions(prices);

        // Check stop loss / take profit
        orders.extend(self.check_exit_conditions(prices));

        // Process new signals
        for signal in signals {
            if signal.strength < self.config.min_signal_strength {
                continue;
            }

            for symbol in &signal.symbols {
                if let Some(order) = self.signal_to_order(signal, symbol, prices) {
                    orders.push(order);
                }
            }
        }

        orders
    }

    /// Convert signal to order
    fn signal_to_order(
        &self,
        signal: &Signal,
        symbol: &str,
        prices: &HashMap<String, f64>,
    ) -> Option<Order> {
        let price = *prices.get(symbol)?;

        let existing_position = self.positions.get(symbol);

        match signal.signal_type {
            SignalType::StrongBuy | SignalType::Buy => {
                // Only buy if we don't have a position or have a short
                if existing_position.is_some() && existing_position.unwrap().is_long() {
                    return None;
                }

                // Check if we can add more positions
                if self.positions.len() >= self.config.max_positions {
                    return None;
                }

                let position_size = self.calculate_position_size(signal, price);
                if position_size > 0.0 {
                    Some(Order {
                        symbol: symbol.to_string(),
                        side: OrderSide::Buy,
                        quantity: position_size,
                        order_type: OrderType::Market,
                        price: None,
                        reason: signal.reason.clone(),
                    })
                } else {
                    None
                }
            }
            SignalType::StrongSell | SignalType::Sell => {
                // Close long position or open short
                if let Some(pos) = existing_position {
                    if pos.is_long() {
                        return Some(Order {
                            symbol: symbol.to_string(),
                            side: OrderSide::Sell,
                            quantity: pos.quantity.abs(),
                            order_type: OrderType::Market,
                            price: None,
                            reason: signal.reason.clone(),
                        });
                    }
                }

                if self.config.allow_short && existing_position.is_none() {
                    let position_size = self.calculate_position_size(signal, price);
                    if position_size > 0.0 {
                        return Some(Order {
                            symbol: symbol.to_string(),
                            side: OrderSide::Sell,
                            quantity: position_size,
                            order_type: OrderType::Market,
                            price: None,
                            reason: signal.reason.clone(),
                        });
                    }
                }

                None
            }
            SignalType::Hold => None,
        }
    }

    /// Calculate position size based on signal and available capital
    fn calculate_position_size(&self, signal: &Signal, price: f64) -> f64 {
        let max_value = self.available_capital * self.config.max_position_size;
        let scaled_value = max_value * signal.strength;
        scaled_value / price
    }

    /// Update positions with new prices
    fn update_positions(&mut self, prices: &HashMap<String, f64>) {
        for (symbol, position) in &mut self.positions {
            if let Some(&price) = prices.get(symbol) {
                position.update_price(price);
            }
        }

        // Recalculate total capital
        let positions_value: f64 = self.positions.values().map(|p| p.value()).sum();
        self.total_capital = self.available_capital + positions_value;
    }

    /// Check stop loss and take profit conditions
    fn check_exit_conditions(&self, prices: &HashMap<String, f64>) -> Vec<Order> {
        let mut orders = Vec::new();

        for (symbol, position) in &self.positions {
            let return_pct = position.return_pct();

            // Stop loss
            if return_pct <= -self.config.stop_loss_pct {
                orders.push(Order {
                    symbol: symbol.clone(),
                    side: if position.is_long() {
                        OrderSide::Sell
                    } else {
                        OrderSide::Buy
                    },
                    quantity: position.quantity.abs(),
                    order_type: OrderType::Market,
                    price: None,
                    reason: "Stop loss triggered".to_string(),
                });
            }
            // Take profit
            else if return_pct >= self.config.take_profit_pct {
                orders.push(Order {
                    symbol: symbol.clone(),
                    side: if position.is_long() {
                        OrderSide::Sell
                    } else {
                        OrderSide::Buy
                    },
                    quantity: position.quantity.abs(),
                    order_type: OrderType::Market,
                    price: None,
                    reason: "Take profit triggered".to_string(),
                });
            }
        }

        orders
    }

    /// Execute an order
    pub fn execute_order(&mut self, order: &Order, fill_price: f64) {
        match order.side {
            OrderSide::Buy => {
                let cost = order.quantity * fill_price;
                if cost <= self.available_capital {
                    self.available_capital -= cost;

                    // Create or update position
                    self.positions.insert(
                        order.symbol.clone(),
                        Position::new(&order.symbol, order.quantity, fill_price),
                    );

                    self.trade_history.push(TradeRecord {
                        symbol: order.symbol.clone(),
                        side: TradeSide::Buy,
                        quantity: order.quantity,
                        price: fill_price,
                        timestamp: chrono::Utc::now().timestamp(),
                        reason: order.reason.clone(),
                    });
                }
            }
            OrderSide::Sell => {
                if let Some(position) = self.positions.remove(&order.symbol) {
                    let proceeds = order.quantity * fill_price;
                    self.available_capital += proceeds;

                    self.trade_history.push(TradeRecord {
                        symbol: order.symbol.clone(),
                        side: TradeSide::Sell,
                        quantity: order.quantity,
                        price: fill_price,
                        timestamp: chrono::Utc::now().timestamp(),
                        reason: order.reason.clone(),
                    });
                }
            }
        }
    }

    /// Get current positions
    pub fn positions(&self) -> &HashMap<String, Position> {
        &self.positions
    }

    /// Get available capital
    pub fn available_capital(&self) -> f64 {
        self.available_capital
    }

    /// Get total capital
    pub fn total_capital(&self) -> f64 {
        self.total_capital
    }

    /// Get trade history
    pub fn trade_history(&self) -> &[TradeRecord] {
        &self.trade_history
    }

    /// Get current return
    pub fn current_return(&self) -> f64 {
        (self.total_capital - self.config.initial_capital) / self.config.initial_capital
    }
}

/// Order to be executed
#[derive(Debug, Clone)]
pub struct Order {
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: f64,
    pub order_type: OrderType,
    pub price: Option<f64>,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Market,
    Limit,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position() {
        let mut pos = Position::new("BTCUSDT", 0.1, 50000.0);
        assert!(pos.is_long());
        assert_eq!(pos.value(), 5000.0);

        pos.update_price(55000.0);
        assert!((pos.return_pct() - 0.1).abs() < 0.001);
        assert!((pos.unrealized_pnl - 500.0).abs() < 0.001);
    }

    #[test]
    fn test_strategy_creation() {
        let config = StrategyConfig::default();
        let strategy = Strategy::new(config);

        assert_eq!(strategy.available_capital(), 10000.0);
        assert_eq!(strategy.total_capital(), 10000.0);
    }
}
