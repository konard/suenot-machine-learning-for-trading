//! Trading module: signal generation and strategy implementation.

pub mod signals;
pub mod strategy;

pub use signals::{SignalDirection, TradingSignal};
pub use strategy::GatedSSMStrategy;
