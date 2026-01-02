//! Trading strategy module
//!
//! Implements signal generation and risk management for ResNet-based trading.

mod risk;
mod signals;

pub use risk::RiskManager;
pub use signals::{TradingSignal, TradingStrategy};
