//! Trading strategy module
//!
//! Implements trading strategies based on Matching Network pattern classification.

mod pattern_strategy;

pub use pattern_strategy::{PatternTradingStrategy, TradingSignal, Position, TradeResult};
