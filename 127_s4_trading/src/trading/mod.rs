//! Trading strategy and signal generation.

pub mod signals;
pub mod strategy;

pub use signals::TradingSignal;
pub use strategy::TradingStrategy;
