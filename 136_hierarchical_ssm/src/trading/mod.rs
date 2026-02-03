//! Trading module: signal generation and strategy execution.

pub mod signals;
pub mod strategy;

pub use signals::{Signal, SignalGenerator};
pub use strategy::{Position, StrategyConfig, TradingStrategy};
