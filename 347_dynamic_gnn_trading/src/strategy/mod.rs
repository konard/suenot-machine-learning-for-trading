//! Trading strategy module
//!
//! This module provides trading signal generation and execution
//! based on Dynamic GNN predictions.

mod signals;
mod execution;

pub use signals::{Signal, SignalType, SignalGenerator};
pub use execution::{Order, OrderSide, OrderType, Position, PositionManager};

use serde::{Deserialize, Serialize};

/// Strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Minimum signal confidence to trade
    pub min_confidence: f64,
    /// Maximum position size as fraction of portfolio
    pub max_position_pct: f64,
    /// Stop loss percentage
    pub stop_loss_pct: f64,
    /// Take profit percentage
    pub take_profit_pct: f64,
    /// Maximum number of positions
    pub max_positions: usize,
    /// Cooldown between trades (seconds)
    pub trade_cooldown: u64,
    /// Use Kelly criterion for sizing
    pub use_kelly: bool,
    /// Kelly fraction (0.5 = half-Kelly)
    pub kelly_fraction: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            max_position_pct: 0.05,
            stop_loss_pct: 0.02,
            take_profit_pct: 0.03,
            max_positions: 5,
            trade_cooldown: 300,
            use_kelly: true,
            kelly_fraction: 0.5,
        }
    }
}

/// Trading strategy using Dynamic GNN
#[derive(Debug)]
pub struct TradingStrategy {
    /// Configuration
    pub config: StrategyConfig,
    /// Signal generator
    signal_generator: SignalGenerator,
    /// Position manager
    position_manager: PositionManager,
    /// Last trade timestamps per symbol
    last_trades: std::collections::HashMap<String, u64>,
}

impl TradingStrategy {
    /// Create a new trading strategy
    pub fn new(config: StrategyConfig) -> Self {
        Self {
            config: config.clone(),
            signal_generator: SignalGenerator::new(),
            position_manager: PositionManager::new(config.max_positions),
            last_trades: std::collections::HashMap::new(),
        }
    }

    /// Process GNN predictions and generate orders
    pub fn process_predictions(
        &mut self,
        symbol: &str,
        price: f64,
        direction_probs: (f64, f64, f64), // (down, neutral, up)
        confidence: f64,
        timestamp: u64,
    ) -> Option<Order> {
        // Check cooldown
        if let Some(&last_trade) = self.last_trades.get(symbol) {
            if timestamp - last_trade < self.config.trade_cooldown * 1000 {
                return None;
            }
        }

        // Check minimum confidence
        if confidence < self.config.min_confidence {
            return None;
        }

        // Generate signal
        let signal = self.signal_generator.generate(
            symbol,
            price,
            direction_probs,
            confidence,
            timestamp,
        );

        // Check if signal is actionable
        if signal.signal_type == SignalType::Hold {
            return None;
        }

        // Check existing position
        let current_position = self.position_manager.get_position(symbol);

        // Determine action
        let order = match (current_position, &signal.signal_type) {
            // No position, go long
            (None, SignalType::Buy) => {
                let size = self.calculate_position_size(price, confidence);
                Some(Order::market(symbol, OrderSide::Buy, size, price))
            }
            // No position, go short
            (None, SignalType::Sell) => {
                let size = self.calculate_position_size(price, confidence);
                Some(Order::market(symbol, OrderSide::Sell, size, price))
            }
            // Have long position, signal to close
            (Some(pos), SignalType::Sell) if pos.side == OrderSide::Buy => {
                Some(Order::market(symbol, OrderSide::Sell, pos.size, price))
            }
            // Have short position, signal to close
            (Some(pos), SignalType::Buy) if pos.side == OrderSide::Sell => {
                Some(Order::market(symbol, OrderSide::Buy, pos.size, price))
            }
            _ => None,
        };

        if order.is_some() {
            self.last_trades.insert(symbol.to_string(), timestamp);
        }

        order
    }

    /// Calculate position size
    fn calculate_position_size(&self, price: f64, confidence: f64) -> f64 {
        let base_size = self.config.max_position_pct;

        if self.config.use_kelly {
            // Kelly criterion: f = (p * b - q) / b
            // where p = win probability, q = 1-p, b = win/loss ratio
            let p = confidence;
            let q = 1.0 - p;
            let b = self.config.take_profit_pct / self.config.stop_loss_pct;

            let kelly = (p * b - q) / b;
            let adjusted_kelly = kelly * self.config.kelly_fraction;

            (base_size * adjusted_kelly.max(0.0).min(1.0)) / price
        } else {
            base_size / price
        }
    }

    /// Check stop loss and take profit
    pub fn check_exits(&mut self, symbol: &str, current_price: f64) -> Option<Order> {
        let position = self.position_manager.get_position(symbol)?;

        let pnl_pct = match position.side {
            OrderSide::Buy => (current_price - position.entry_price) / position.entry_price,
            OrderSide::Sell => (position.entry_price - current_price) / position.entry_price,
        };

        // Check stop loss
        if pnl_pct <= -self.config.stop_loss_pct {
            let close_side = match position.side {
                OrderSide::Buy => OrderSide::Sell,
                OrderSide::Sell => OrderSide::Buy,
            };
            return Some(Order::market(symbol, close_side, position.size, current_price));
        }

        // Check take profit
        if pnl_pct >= self.config.take_profit_pct {
            let close_side = match position.side {
                OrderSide::Buy => OrderSide::Sell,
                OrderSide::Sell => OrderSide::Buy,
            };
            return Some(Order::market(symbol, close_side, position.size, current_price));
        }

        None
    }

    /// Execute an order (update internal state)
    pub fn execute_order(&mut self, order: &Order, fill_price: f64, timestamp: u64) {
        // Update position manager
        if let Some(current) = self.position_manager.get_position(&order.symbol) {
            // Closing position
            if (order.side == OrderSide::Sell && current.side == OrderSide::Buy)
                || (order.side == OrderSide::Buy && current.side == OrderSide::Sell)
            {
                self.position_manager.close_position(&order.symbol);
            }
        } else {
            // Opening new position
            self.position_manager.open_position(
                &order.symbol,
                order.side.clone(),
                order.size,
                fill_price,
                timestamp,
            );
        }
    }

    /// Get current positions
    pub fn positions(&self) -> Vec<&Position> {
        self.position_manager.all_positions()
    }

    /// Get portfolio stats
    pub fn portfolio_stats(&self, prices: &std::collections::HashMap<String, f64>) -> PortfolioStats {
        let positions = self.positions();
        let mut total_pnl = 0.0;
        let mut total_value = 0.0;

        for pos in &positions {
            if let Some(&current_price) = prices.get(&pos.symbol) {
                let pnl = match pos.side {
                    OrderSide::Buy => (current_price - pos.entry_price) * pos.size,
                    OrderSide::Sell => (pos.entry_price - current_price) * pos.size,
                };
                total_pnl += pnl;
                total_value += pos.size * current_price;
            }
        }

        PortfolioStats {
            num_positions: positions.len(),
            total_value,
            unrealized_pnl: total_pnl,
            pnl_pct: if total_value > 0.0 {
                total_pnl / total_value
            } else {
                0.0
            },
        }
    }
}

/// Portfolio statistics
#[derive(Debug, Clone)]
pub struct PortfolioStats {
    /// Number of open positions
    pub num_positions: usize,
    /// Total position value
    pub total_value: f64,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
    /// PnL as percentage
    pub pnl_pct: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_creation() {
        let config = StrategyConfig::default();
        let strategy = TradingStrategy::new(config);
        assert!(strategy.positions().is_empty());
    }

    #[test]
    fn test_signal_processing() {
        let config = StrategyConfig::default();
        let mut strategy = TradingStrategy::new(config);

        // Strong buy signal
        let order = strategy.process_predictions(
            "BTCUSDT",
            50000.0,
            (0.1, 0.2, 0.7), // 70% up probability
            0.8,             // 80% confidence
            1000,
        );

        assert!(order.is_some());
        let order = order.unwrap();
        assert_eq!(order.side, OrderSide::Buy);
    }

    #[test]
    fn test_cooldown() {
        let mut config = StrategyConfig::default();
        config.trade_cooldown = 60;
        let mut strategy = TradingStrategy::new(config);

        // First trade
        let order1 = strategy.process_predictions(
            "BTCUSDT",
            50000.0,
            (0.1, 0.2, 0.7),
            0.8,
            1000,
        );
        assert!(order1.is_some());

        // Second trade too soon (within cooldown)
        let order2 = strategy.process_predictions(
            "BTCUSDT",
            50100.0,
            (0.1, 0.2, 0.7),
            0.8,
            30000, // 30 seconds later
        );
        assert!(order2.is_none()); // Should be blocked by cooldown
    }
}
