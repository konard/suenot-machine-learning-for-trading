//! Trading signal generation and strategy execution.

mod signals;
mod strategy;

pub use signals::{Signal, SignalGenerator};
pub use strategy::{Position, TradingStrategy};
