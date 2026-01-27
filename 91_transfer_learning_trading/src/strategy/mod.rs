//! Trading strategy module
//!
//! Implements trading strategies based on transfer learning predictions.
//! Includes signal generation, position management, and risk controls.

mod transfer_strategy;

pub use transfer_strategy::{TransferTradingStrategy, TradingSignal, Position, TradeResult};
