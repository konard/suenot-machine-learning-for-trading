//! Order execution and position management

use std::collections::HashMap;
use super::{Signal, SignalType};

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Order for execution
#[derive(Debug, Clone)]
pub struct Order {
    /// Symbol
    pub symbol: String,
    /// Side
    pub side: OrderSide,
    /// Size
    pub size: f64,
    /// Price (None for market order)
    pub price: Option<f64>,
    /// Stop loss price
    pub stop_loss: Option<f64>,
    /// Take profit price
    pub take_profit: Option<f64>,
}

impl Order {
    /// Create a new market order
    pub fn market(symbol: impl Into<String>, side: OrderSide, size: f64) -> Self {
        Self {
            symbol: symbol.into(),
            side,
            size,
            price: None,
            stop_loss: None,
            take_profit: None,
        }
    }

    /// Create a new limit order
    pub fn limit(symbol: impl Into<String>, side: OrderSide, size: f64, price: f64) -> Self {
        Self {
            symbol: symbol.into(),
            side,
            size,
            price: Some(price),
            stop_loss: None,
            take_profit: None,
        }
    }

    /// Set stop loss
    pub fn with_stop_loss(mut self, price: f64) -> Self {
        self.stop_loss = Some(price);
        self
    }

    /// Set take profit
    pub fn with_take_profit(mut self, price: f64) -> Self {
        self.take_profit = Some(price);
        self
    }
}

/// Current position
#[derive(Debug, Clone)]
pub struct Position {
    /// Symbol
    pub symbol: String,
    /// Size (positive for long, negative for short)
    pub size: f64,
    /// Entry price
    pub entry_price: f64,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
    /// Entry timestamp
    pub entry_time: u64,
}

impl Position {
    /// Create a new position
    pub fn new(symbol: impl Into<String>, size: f64, entry_price: f64) -> Self {
        Self {
            symbol: symbol.into(),
            size,
            entry_price,
            unrealized_pnl: 0.0,
            entry_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        }
    }

    /// Update unrealized PnL
    pub fn update_pnl(&mut self, current_price: f64) {
        self.unrealized_pnl = self.size * (current_price - self.entry_price);
    }

    /// Check if long position
    pub fn is_long(&self) -> bool {
        self.size > 0.0
    }

    /// Check if short position
    pub fn is_short(&self) -> bool {
        self.size < 0.0
    }

    /// Get return percentage
    pub fn return_pct(&self, current_price: f64) -> f64 {
        if self.entry_price > 0.0 {
            (current_price / self.entry_price - 1.0) * self.size.signum() * 100.0
        } else {
            0.0
        }
    }
}

/// Strategy configuration
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Maximum position size per symbol (as fraction of portfolio)
    pub max_position_size: f64,
    /// Maximum total exposure
    pub max_total_exposure: f64,
    /// Maximum leverage
    pub max_leverage: f64,
    /// Stop loss percentage
    pub stop_loss_pct: f64,
    /// Take profit percentage
    pub take_profit_pct: f64,
    /// Minimum signal strength to trade
    pub min_signal_strength: f64,
    /// Minimum confidence to trade
    pub min_confidence: f64,
    /// Symbols to trade
    pub symbols: Vec<String>,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            max_position_size: 0.05,  // 5% max per position
            max_total_exposure: 0.5,  // 50% max total
            max_leverage: 3.0,
            stop_loss_pct: 0.02,  // 2% stop loss
            take_profit_pct: 0.04,  // 4% take profit
            min_signal_strength: 0.3,
            min_confidence: 0.5,
            symbols: vec![
                "BTCUSDT".to_string(),
                "ETHUSDT".to_string(),
                "SOLUSDT".to_string(),
            ],
        }
    }
}

/// Trading strategy
#[derive(Debug)]
pub struct TradingStrategy {
    /// Configuration
    config: StrategyConfig,
    /// Current positions
    positions: HashMap<String, Position>,
    /// Portfolio value
    portfolio_value: f64,
    /// Current prices
    prices: HashMap<String, f64>,
}

impl TradingStrategy {
    /// Create a new trading strategy
    pub fn new(config: StrategyConfig, portfolio_value: f64) -> Self {
        Self {
            config,
            positions: HashMap::new(),
            portfolio_value,
            prices: HashMap::new(),
        }
    }

    /// Update current price
    pub fn update_price(&mut self, symbol: &str, price: f64) {
        self.prices.insert(symbol.to_string(), price);

        // Update position PnL
        if let Some(pos) = self.positions.get_mut(symbol) {
            pos.update_pnl(price);
        }
    }

    /// Generate orders from signals
    pub fn generate_orders(&self, signals: &[Signal]) -> Vec<Order> {
        let mut orders = Vec::new();

        for signal in signals {
            // Check minimum requirements
            if signal.strength < self.config.min_signal_strength {
                continue;
            }
            if signal.confidence < self.config.min_confidence {
                continue;
            }

            // Get current position
            let current_pos = self.positions.get(&signal.symbol);
            let current_size = current_pos.map(|p| p.size).unwrap_or(0.0);

            // Get current price
            let price = match self.prices.get(&signal.symbol) {
                Some(&p) => p,
                None => continue,
            };

            // Calculate target size based on signal
            let target_size = self.calculate_target_size(signal, price);

            // Generate order if needed
            if let Some(order) = self.create_order(&signal.symbol, current_size, target_size, price) {
                orders.push(order);
            }
        }

        orders
    }

    /// Calculate target position size
    fn calculate_target_size(&self, signal: &Signal, price: f64) -> f64 {
        // Kelly-inspired sizing
        let edge = signal.strength * signal.confidence;
        let kelly_fraction = edge.max(0.0).min(0.25);  // Cap at 25%

        // Position size in quote currency
        let position_value = self.portfolio_value * kelly_fraction * self.config.max_position_size;

        // Size in base currency
        let size = position_value / price;

        // Apply direction
        match signal.signal_type {
            SignalType::StrongBuy | SignalType::Buy => size,
            SignalType::StrongSell | SignalType::Sell => -size,
            SignalType::Neutral => 0.0,
        }
    }

    /// Create order to reach target size
    fn create_order(&self, symbol: &str, current: f64, target: f64, price: f64) -> Option<Order> {
        let diff = target - current;

        // Minimum order threshold
        if diff.abs() * price < 10.0 {  // Less than $10
            return None;
        }

        let (side, size) = if diff > 0.0 {
            (OrderSide::Buy, diff)
        } else {
            (OrderSide::Sell, -diff)
        };

        let mut order = Order::market(symbol, side, size);

        // Add stop loss and take profit
        if target != 0.0 {
            if target > 0.0 {
                order.stop_loss = Some(price * (1.0 - self.config.stop_loss_pct));
                order.take_profit = Some(price * (1.0 + self.config.take_profit_pct));
            } else {
                order.stop_loss = Some(price * (1.0 + self.config.stop_loss_pct));
                order.take_profit = Some(price * (1.0 - self.config.take_profit_pct));
            }
        }

        Some(order)
    }

    /// Execute an order (simulated)
    pub fn execute_order(&mut self, order: &Order, fill_price: f64) {
        let size_delta = if order.side == OrderSide::Buy {
            order.size
        } else {
            -order.size
        };

        // Update or create position
        if let Some(pos) = self.positions.get_mut(&order.symbol) {
            pos.size += size_delta;
            if pos.size.abs() < 1e-10 {
                self.positions.remove(&order.symbol);
            }
        } else if size_delta.abs() > 1e-10 {
            self.positions.insert(
                order.symbol.clone(),
                Position::new(&order.symbol, size_delta, fill_price),
            );
        }
    }

    /// Get current positions
    pub fn positions(&self) -> &HashMap<String, Position> {
        &self.positions
    }

    /// Get total exposure
    pub fn total_exposure(&self) -> f64 {
        self.positions
            .iter()
            .map(|(symbol, pos)| {
                let price = self.prices.get(symbol).copied().unwrap_or(0.0);
                pos.size.abs() * price
            })
            .sum::<f64>()
            / self.portfolio_value
    }

    /// Get total PnL
    pub fn total_pnl(&self) -> f64 {
        self.positions.values().map(|p| p.unrealized_pnl).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_creation() {
        let order = Order::market("BTCUSDT", OrderSide::Buy, 0.1)
            .with_stop_loss(49000.0)
            .with_take_profit(55000.0);

        assert_eq!(order.symbol, "BTCUSDT");
        assert_eq!(order.side, OrderSide::Buy);
        assert_eq!(order.stop_loss, Some(49000.0));
    }

    #[test]
    fn test_position() {
        let mut pos = Position::new("BTCUSDT", 0.1, 50000.0);
        pos.update_pnl(52000.0);

        assert!(pos.is_long());
        assert_eq!(pos.unrealized_pnl, 200.0);  // 0.1 * 2000
    }

    #[test]
    fn test_strategy() {
        let config = StrategyConfig::default();
        let mut strategy = TradingStrategy::new(config, 100000.0);

        strategy.update_price("BTCUSDT", 50000.0);

        let signal = Signal::new("BTCUSDT", SignalType::Buy, 0.7)
            .with_confidence(0.8);

        let orders = strategy.generate_orders(&[signal]);
        assert!(!orders.is_empty());
    }
}
