//! Simulated broker for backtesting.

use crate::models::{Order, OrderSide, OrderStatus, OrderType, Position};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Configuration for the simulated broker.
#[derive(Debug, Clone)]
pub struct BrokerConfig {
    /// Initial cash balance
    pub initial_cash: f64,
    /// Trading fee as a percentage (e.g., 0.001 = 0.1%)
    pub fee_rate: f64,
    /// Slippage as a percentage (e.g., 0.0005 = 0.05%)
    pub slippage: f64,
    /// Enable margin trading
    pub margin_enabled: bool,
    /// Maximum leverage
    pub max_leverage: f64,
}

impl Default for BrokerConfig {
    fn default() -> Self {
        Self {
            initial_cash: 10000.0,
            fee_rate: 0.001,    // 0.1% (typical Bybit maker fee)
            slippage: 0.0005,  // 0.05%
            margin_enabled: false,
            max_leverage: 1.0,
        }
    }
}

/// Simulated broker that executes orders and tracks positions.
#[derive(Debug)]
pub struct SimulatedBroker {
    config: BrokerConfig,
    cash: f64,
    positions: HashMap<String, Position>,
    orders: Vec<Order>,
    order_counter: u64,
    total_fees_paid: f64,
    equity_history: Vec<(DateTime<Utc>, f64)>,
}

impl SimulatedBroker {
    /// Create a new simulated broker.
    pub fn new(config: BrokerConfig) -> Self {
        Self {
            cash: config.initial_cash,
            config,
            positions: HashMap::new(),
            orders: Vec::new(),
            order_counter: 0,
            total_fees_paid: 0.0,
            equity_history: Vec::new(),
        }
    }

    /// Get current cash balance.
    pub fn cash(&self) -> f64 {
        self.cash
    }

    /// Get current positions.
    pub fn positions(&self) -> &HashMap<String, Position> {
        &self.positions
    }

    /// Get position for a specific symbol.
    pub fn get_position(&self, symbol: &str) -> Option<&Position> {
        self.positions.get(symbol)
    }

    /// Get total fees paid.
    pub fn total_fees(&self) -> f64 {
        self.total_fees_paid
    }

    /// Get equity history.
    pub fn equity_history(&self) -> &[(DateTime<Utc>, f64)] {
        &self.equity_history
    }

    /// Calculate total equity (cash + position values).
    pub fn equity(&self, prices: &HashMap<String, f64>) -> f64 {
        let position_value: f64 = self
            .positions
            .iter()
            .filter_map(|(symbol, pos)| {
                prices.get(symbol).map(|&price| pos.value(price))
            })
            .sum();

        self.cash + position_value
    }

    /// Submit a new order.
    pub fn submit_order(&mut self, mut order: Order) -> String {
        self.order_counter += 1;
        order.id = format!("ORD-{:06}", self.order_counter);
        let id = order.id.clone();
        self.orders.push(order);
        id
    }

    /// Submit a market buy order.
    pub fn buy(&mut self, symbol: &str, quantity: f64) -> String {
        let order = Order::market(
            String::new(),
            symbol.to_string(),
            OrderSide::Buy,
            quantity,
        );
        self.submit_order(order)
    }

    /// Submit a market sell order.
    pub fn sell(&mut self, symbol: &str, quantity: f64) -> String {
        let order = Order::market(
            String::new(),
            symbol.to_string(),
            OrderSide::Sell,
            quantity,
        );
        self.submit_order(order)
    }

    /// Close all positions at current prices.
    pub fn close_all_positions(&mut self, prices: &HashMap<String, f64>) {
        let positions_to_close: Vec<(String, f64, OrderSide)> = self
            .positions
            .iter()
            .filter_map(|(symbol, pos)| {
                if pos.size > 0.0 {
                    Some((symbol.clone(), pos.size, pos.side))
                } else {
                    None
                }
            })
            .collect();

        for (symbol, size, side) in positions_to_close {
            if let Some(&price) = prices.get(&symbol) {
                let close_side = match side {
                    OrderSide::Buy => OrderSide::Sell,
                    OrderSide::Sell => OrderSide::Buy,
                };
                let order = Order::market(String::new(), symbol, close_side, size);
                let order_id = self.submit_order(order);
                self.process_order(&order_id, price);
            }
        }
    }

    /// Process pending orders at the given price.
    pub fn process_orders(&mut self, symbol: &str, price: f64) {
        let order_ids: Vec<String> = self
            .orders
            .iter()
            .filter(|o| o.symbol == symbol && o.is_active())
            .map(|o| o.id.clone())
            .collect();

        for order_id in order_ids {
            self.process_order(&order_id, price);
        }
    }

    /// Process a specific order.
    fn process_order(&mut self, order_id: &str, market_price: f64) {
        // Find order index to avoid borrow issues
        let order_idx = match self.orders.iter().position(|o| o.id == order_id) {
            Some(idx) => idx,
            None => return,
        };

        // Check if order is active and get execution info
        let (should_execute, exec_price) = {
            let order = &self.orders[order_idx];
            if !order.is_active() {
                return;
            }

            match order.order_type {
                OrderType::Market => (true, market_price),
                OrderType::Limit => {
                    if let Some(limit_price) = order.price {
                        let should_fill = match order.side {
                            OrderSide::Buy => market_price <= limit_price,
                            OrderSide::Sell => market_price >= limit_price,
                        };
                        (should_fill, limit_price)
                    } else {
                        (false, 0.0)
                    }
                }
                _ => (false, 0.0),
            }
        };

        if should_execute {
            self.execute_market_order_by_idx(order_idx, exec_price);
        }
    }

    /// Execute a market order by index.
    fn execute_market_order_by_idx(&mut self, order_idx: usize, market_price: f64) {
        // Extract order data first
        let (side, quantity, symbol) = {
            let order = &self.orders[order_idx];
            (order.side, order.quantity, order.symbol.clone())
        };

        // Apply slippage
        let fill_price = match side {
            OrderSide::Buy => market_price * (1.0 + self.config.slippage),
            OrderSide::Sell => market_price * (1.0 - self.config.slippage),
        };

        let trade_value = fill_price * quantity;
        let fee = trade_value * self.config.fee_rate;

        match side {
            OrderSide::Buy => {
                let total_cost = trade_value + fee;
                if self.cash < total_cost && !self.config.margin_enabled {
                    self.orders[order_idx].status = OrderStatus::Rejected;
                    return;
                }

                self.cash -= total_cost;
                self.total_fees_paid += fee;

                // Update or create position
                if let Some(pos) = self.positions.get_mut(&symbol) {
                    if pos.side == OrderSide::Buy {
                        pos.add(quantity, fill_price);
                    } else {
                        // Closing short position
                        let pnl = pos.reduce(quantity, fill_price);
                        self.cash += pnl;
                        if pos.is_closed() {
                            self.positions.remove(&symbol);
                        }
                    }
                } else {
                    let pos = Position::new(
                        symbol.clone(),
                        OrderSide::Buy,
                        quantity,
                        fill_price,
                    );
                    self.positions.insert(symbol, pos);
                }
            }
            OrderSide::Sell => {
                // Check if we have a long position to sell
                let should_remove = if let Some(pos) = self.positions.get_mut(&symbol) {
                    if pos.side == OrderSide::Buy {
                        let sell_qty = quantity.min(pos.size);
                        let pnl = pos.reduce(sell_qty, fill_price);
                        self.cash += trade_value - fee + pnl;
                        self.total_fees_paid += fee;
                        pos.is_closed()
                    } else {
                        false
                    }
                } else if self.config.margin_enabled {
                    // Open short position
                    self.cash += trade_value - fee;
                    self.total_fees_paid += fee;
                    let pos = Position::new(
                        symbol.clone(),
                        OrderSide::Sell,
                        quantity,
                        fill_price,
                    );
                    self.positions.insert(symbol.clone(), pos);
                    false
                } else {
                    self.orders[order_idx].status = OrderStatus::Rejected;
                    return;
                };

                if should_remove {
                    self.positions.remove(&symbol);
                }
            }
        }

        self.orders[order_idx].fill(fill_price, quantity);
    }

    /// Record current equity for history tracking.
    pub fn record_equity(&mut self, timestamp: DateTime<Utc>, prices: &HashMap<String, f64>) {
        let eq = self.equity(prices);
        self.equity_history.push((timestamp, eq));
    }

    /// Get filled orders.
    pub fn filled_orders(&self) -> Vec<&Order> {
        self.orders.iter().filter(|o| o.is_filled()).collect()
    }

    /// Get all orders.
    pub fn all_orders(&self) -> &[Order] {
        &self.orders
    }

    /// Reset broker to initial state.
    pub fn reset(&mut self) {
        self.cash = self.config.initial_cash;
        self.positions.clear();
        self.orders.clear();
        self.order_counter = 0;
        self.total_fees_paid = 0.0;
        self.equity_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_buy() {
        let config = BrokerConfig {
            initial_cash: 10000.0,
            fee_rate: 0.001,
            slippage: 0.0,
            ..Default::default()
        };

        let mut broker = SimulatedBroker::new(config);
        let order_id = broker.buy("BTCUSDT", 0.1);
        broker.process_order(&order_id, 50000.0);

        assert!(broker.positions.contains_key("BTCUSDT"));
        let pos = broker.positions.get("BTCUSDT").unwrap();
        assert!((pos.size - 0.1).abs() < 1e-10);
        assert!(broker.cash < 10000.0);
    }

    #[test]
    fn test_round_trip() {
        let config = BrokerConfig {
            initial_cash: 10000.0,
            fee_rate: 0.001,
            slippage: 0.0,
            ..Default::default()
        };

        let mut broker = SimulatedBroker::new(config);

        // Buy
        let order_id = broker.buy("BTCUSDT", 0.1);
        broker.process_order(&order_id, 50000.0);

        // Sell at higher price
        let order_id = broker.sell("BTCUSDT", 0.1);
        broker.process_order(&order_id, 55000.0);

        // Should have profit minus fees
        assert!(broker.cash > 10000.0);
        assert!(broker.positions.is_empty() || broker.positions.get("BTCUSDT").map(|p| p.is_closed()).unwrap_or(true));
    }
}
