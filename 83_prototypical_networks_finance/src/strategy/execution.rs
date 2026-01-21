//! Position management and execution
//!
//! Handles position tracking and execution logic.

use crate::strategy::signals::{TradingSignal, SignalType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Position side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionSide {
    Long,
    Short,
    Flat,
}

/// Represents a trading position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Position side
    pub side: PositionSide,
    /// Entry price
    pub entry_price: f64,
    /// Position size (in base currency units)
    pub size: f64,
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
    /// Current mark price
    pub mark_price: f64,
}

impl Position {
    /// Create a new flat position
    pub fn flat() -> Self {
        Self {
            side: PositionSide::Flat,
            entry_price: 0.0,
            size: 0.0,
            entry_time: Utc::now(),
            unrealized_pnl: 0.0,
            mark_price: 0.0,
        }
    }

    /// Create a new long position
    pub fn long(entry_price: f64, size: f64) -> Self {
        Self {
            side: PositionSide::Long,
            entry_price,
            size,
            entry_time: Utc::now(),
            unrealized_pnl: 0.0,
            mark_price: entry_price,
        }
    }

    /// Create a new short position
    pub fn short(entry_price: f64, size: f64) -> Self {
        Self {
            side: PositionSide::Short,
            entry_price,
            size,
            entry_time: Utc::now(),
            unrealized_pnl: 0.0,
            mark_price: entry_price,
        }
    }

    /// Update the position with current market price
    pub fn update_price(&mut self, current_price: f64) {
        self.mark_price = current_price;
        self.unrealized_pnl = match self.side {
            PositionSide::Long => (current_price - self.entry_price) * self.size,
            PositionSide::Short => (self.entry_price - current_price) * self.size,
            PositionSide::Flat => 0.0,
        };
    }

    /// Calculate return percentage
    pub fn return_pct(&self) -> f64 {
        if self.entry_price > 0.0 && self.size > 0.0 {
            match self.side {
                PositionSide::Long => (self.mark_price / self.entry_price - 1.0) * 100.0,
                PositionSide::Short => (1.0 - self.mark_price / self.entry_price) * 100.0,
                PositionSide::Flat => 0.0,
            }
        } else {
            0.0
        }
    }

    /// Check if position is profitable
    pub fn is_profitable(&self) -> bool {
        self.unrealized_pnl > 0.0
    }

    /// Check if position is open
    pub fn is_open(&self) -> bool {
        !matches!(self.side, PositionSide::Flat)
    }
}

/// Configuration for position management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    /// Maximum position size (in quote currency)
    pub max_position_size: f64,
    /// Minimum position size
    pub min_position_size: f64,
    /// Stop loss percentage (e.g., 0.02 = 2%)
    pub stop_loss_pct: f64,
    /// Take profit percentage
    pub take_profit_pct: f64,
    /// Whether to use trailing stop
    pub use_trailing_stop: bool,
    /// Trailing stop activation percentage
    pub trailing_activation_pct: f64,
    /// Trailing stop distance percentage
    pub trailing_distance_pct: f64,
    /// Maximum drawdown allowed before reducing position
    pub max_drawdown_pct: f64,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            max_position_size: 10000.0,
            min_position_size: 10.0,
            stop_loss_pct: 0.02,
            take_profit_pct: 0.05,
            use_trailing_stop: true,
            trailing_activation_pct: 0.02,
            trailing_distance_pct: 0.01,
            max_drawdown_pct: 0.10,
        }
    }
}

/// Order type for execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    MarketOpen,
    MarketClose,
    LimitOpen,
    LimitClose,
    StopLoss,
    TakeProfit,
}

/// Order to be executed
#[derive(Debug, Clone)]
pub struct Order {
    pub order_type: OrderType,
    pub side: PositionSide,
    pub size: f64,
    pub price: Option<f64>,
    pub reason: String,
}

/// Position manager for tracking and managing trades
pub struct PositionManager {
    config: ExecutionConfig,
    current_position: Position,
    /// Highest PnL achieved (for trailing stop)
    peak_pnl: f64,
    /// Total realized PnL
    realized_pnl: f64,
    /// Trade count
    trade_count: usize,
    /// Winning trades
    winning_trades: usize,
}

impl PositionManager {
    /// Create a new position manager
    pub fn new(config: ExecutionConfig) -> Self {
        Self {
            config,
            current_position: Position::flat(),
            peak_pnl: 0.0,
            realized_pnl: 0.0,
            trade_count: 0,
            winning_trades: 0,
        }
    }

    /// Process a trading signal and generate orders
    pub fn process_signal(
        &mut self,
        signal: &TradingSignal,
        current_price: f64,
        available_capital: f64,
    ) -> Vec<Order> {
        let mut orders = Vec::new();

        // Update position with current price
        self.current_position.update_price(current_price);

        // Check for stop loss and take profit
        if self.current_position.is_open() {
            if let Some(stop_order) = self.check_stop_loss(current_price) {
                orders.push(stop_order);
                return orders; // Exit immediately on stop loss
            }

            if let Some(tp_order) = self.check_take_profit(current_price) {
                orders.push(tp_order);
                return orders; // Exit on take profit
            }

            // Update trailing stop
            if self.config.use_trailing_stop {
                self.update_trailing_stop();
            }
        }

        // Process signal for position changes
        match signal.signal_type {
            SignalType::StrongBuy | SignalType::Buy => {
                orders.extend(self.handle_buy_signal(signal, current_price, available_capital));
            }
            SignalType::StrongSell | SignalType::Sell => {
                orders.extend(self.handle_sell_signal(signal, current_price, available_capital));
            }
            SignalType::Hold => {
                // No new orders, but might close position if unusual
                if signal.is_unusual && self.current_position.is_open() {
                    orders.push(self.close_position_order("Unusual market conditions"));
                }
            }
        }

        orders
    }

    /// Handle buy signal
    fn handle_buy_signal(
        &mut self,
        signal: &TradingSignal,
        current_price: f64,
        available_capital: f64,
    ) -> Vec<Order> {
        let mut orders = Vec::new();

        match self.current_position.side {
            PositionSide::Short => {
                // Close short position first
                orders.push(self.close_position_order("Signal reversal to long"));
            }
            PositionSide::Flat => {
                // Open new long position
                let size = self.calculate_position_size(signal, available_capital, current_price);
                if size >= self.config.min_position_size {
                    orders.push(Order {
                        order_type: OrderType::MarketOpen,
                        side: PositionSide::Long,
                        size,
                        price: Some(current_price),
                        reason: signal.reason.clone(),
                    });
                }
            }
            PositionSide::Long => {
                // Already long, maybe adjust position size
                // For simplicity, we don't adjust existing positions
            }
        }

        orders
    }

    /// Handle sell signal
    fn handle_sell_signal(
        &mut self,
        signal: &TradingSignal,
        current_price: f64,
        available_capital: f64,
    ) -> Vec<Order> {
        let mut orders = Vec::new();

        match self.current_position.side {
            PositionSide::Long => {
                // Close long position first
                orders.push(self.close_position_order("Signal reversal to short"));
            }
            PositionSide::Flat => {
                // Open new short position
                let size = self.calculate_position_size(signal, available_capital, current_price);
                if size >= self.config.min_position_size {
                    orders.push(Order {
                        order_type: OrderType::MarketOpen,
                        side: PositionSide::Short,
                        size,
                        price: Some(current_price),
                        reason: signal.reason.clone(),
                    });
                }
            }
            PositionSide::Short => {
                // Already short
            }
        }

        orders
    }

    /// Calculate position size based on signal and config
    fn calculate_position_size(
        &self,
        signal: &TradingSignal,
        available_capital: f64,
        current_price: f64,
    ) -> f64 {
        let target_notional = signal.position_size * available_capital;
        let capped_notional = target_notional.min(self.config.max_position_size);
        let size = capped_notional / current_price;

        size.max(0.0)
    }

    /// Create order to close current position
    fn close_position_order(&self, reason: &str) -> Order {
        Order {
            order_type: OrderType::MarketClose,
            side: PositionSide::Flat,
            size: self.current_position.size,
            price: None,
            reason: reason.to_string(),
        }
    }

    /// Check for stop loss trigger
    fn check_stop_loss(&self, current_price: f64) -> Option<Order> {
        if !self.current_position.is_open() {
            return None;
        }

        let loss_pct = match self.current_position.side {
            PositionSide::Long => {
                (self.current_position.entry_price - current_price) / self.current_position.entry_price
            }
            PositionSide::Short => {
                (current_price - self.current_position.entry_price) / self.current_position.entry_price
            }
            PositionSide::Flat => return None,
        };

        if loss_pct >= self.config.stop_loss_pct {
            Some(Order {
                order_type: OrderType::StopLoss,
                side: PositionSide::Flat,
                size: self.current_position.size,
                price: Some(current_price),
                reason: format!("Stop loss triggered at {:.2}% loss", loss_pct * 100.0),
            })
        } else {
            None
        }
    }

    /// Check for take profit trigger
    fn check_take_profit(&self, current_price: f64) -> Option<Order> {
        if !self.current_position.is_open() {
            return None;
        }

        let profit_pct = match self.current_position.side {
            PositionSide::Long => {
                (current_price - self.current_position.entry_price) / self.current_position.entry_price
            }
            PositionSide::Short => {
                (self.current_position.entry_price - current_price) / self.current_position.entry_price
            }
            PositionSide::Flat => return None,
        };

        if profit_pct >= self.config.take_profit_pct {
            Some(Order {
                order_type: OrderType::TakeProfit,
                side: PositionSide::Flat,
                size: self.current_position.size,
                price: Some(current_price),
                reason: format!("Take profit triggered at {:.2}% profit", profit_pct * 100.0),
            })
        } else {
            None
        }
    }

    /// Update trailing stop
    fn update_trailing_stop(&mut self) {
        if self.current_position.unrealized_pnl > self.peak_pnl {
            self.peak_pnl = self.current_position.unrealized_pnl;
        }
    }

    /// Execute an order and update position
    pub fn execute_order(&mut self, order: &Order, execution_price: f64) {
        match order.order_type {
            OrderType::MarketOpen | OrderType::LimitOpen => {
                self.current_position = match order.side {
                    PositionSide::Long => Position::long(execution_price, order.size),
                    PositionSide::Short => Position::short(execution_price, order.size),
                    PositionSide::Flat => Position::flat(),
                };
                self.peak_pnl = 0.0;
            }
            OrderType::MarketClose | OrderType::LimitClose | OrderType::StopLoss | OrderType::TakeProfit => {
                // Close position and record PnL
                self.current_position.update_price(execution_price);
                self.realized_pnl += self.current_position.unrealized_pnl;
                self.trade_count += 1;
                if self.current_position.is_profitable() {
                    self.winning_trades += 1;
                }
                self.current_position = Position::flat();
                self.peak_pnl = 0.0;
            }
        }
    }

    /// Get current position
    pub fn position(&self) -> &Position {
        &self.current_position
    }

    /// Get realized PnL
    pub fn realized_pnl(&self) -> f64 {
        self.realized_pnl
    }

    /// Get total PnL (realized + unrealized)
    pub fn total_pnl(&self) -> f64 {
        self.realized_pnl + self.current_position.unrealized_pnl
    }

    /// Get win rate
    pub fn win_rate(&self) -> f64 {
        if self.trade_count > 0 {
            self.winning_trades as f64 / self.trade_count as f64
        } else {
            0.0
        }
    }

    /// Get trade count
    pub fn trade_count(&self) -> usize {
        self.trade_count
    }
}

impl Default for PositionManager {
    fn default() -> Self {
        Self::new(ExecutionConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::MarketRegime;

    fn create_test_signal(signal_type: SignalType, confidence: f64) -> TradingSignal {
        TradingSignal {
            signal_type,
            regime: MarketRegime::StrongUptrend,
            confidence,
            position_size: 0.5,
            is_unusual: false,
            reason: "Test signal".to_string(),
        }
    }

    #[test]
    fn test_position_creation() {
        let pos = Position::long(100.0, 1.0);
        assert_eq!(pos.side, PositionSide::Long);
        assert_eq!(pos.entry_price, 100.0);
        assert!(pos.is_open());
    }

    #[test]
    fn test_position_update() {
        let mut pos = Position::long(100.0, 1.0);
        pos.update_price(110.0);

        assert_eq!(pos.unrealized_pnl, 10.0);
        assert!(pos.is_profitable());
        assert!((pos.return_pct() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_short_position() {
        let mut pos = Position::short(100.0, 1.0);
        pos.update_price(90.0);

        assert_eq!(pos.unrealized_pnl, 10.0);
        assert!(pos.is_profitable());
    }

    #[test]
    fn test_position_manager_buy_signal() {
        let mut manager = PositionManager::new(ExecutionConfig::default());
        let signal = create_test_signal(SignalType::Buy, 0.8);

        let orders = manager.process_signal(&signal, 100.0, 10000.0);

        assert!(!orders.is_empty());
        assert_eq!(orders[0].side, PositionSide::Long);
    }

    #[test]
    fn test_stop_loss() {
        let config = ExecutionConfig {
            stop_loss_pct: 0.02,
            ..Default::default()
        };
        let mut manager = PositionManager::new(config);

        // Open a long position
        let signal = create_test_signal(SignalType::Buy, 0.8);
        let orders = manager.process_signal(&signal, 100.0, 10000.0);
        manager.execute_order(&orders[0], 100.0);

        // Price drops 3% - should trigger stop loss
        let hold_signal = TradingSignal {
            signal_type: SignalType::Hold,
            regime: MarketRegime::Sideways,
            confidence: 0.5,
            position_size: 0.0,
            is_unusual: false,
            reason: "Hold".to_string(),
        };
        let orders = manager.process_signal(&hold_signal, 97.0, 10000.0);

        assert!(!orders.is_empty());
        assert_eq!(orders[0].order_type, OrderType::StopLoss);
    }
}
