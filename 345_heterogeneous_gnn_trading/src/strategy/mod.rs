//! Trading strategy module
//!
//! Provides signal generation and execution components for trading.

mod signals;
mod execution;

pub use signals::{Signal, SignalType, MetapathSignal, SignalAggregator};
pub use execution::{TradingStrategy, StrategyConfig, Position, Order, OrderSide};
