//! Trading Strategy Module
//!
//! Implements trading strategies based on neural network predictions

mod signals;
mod position;
mod trading;

pub use signals::{Signal, SignalGenerator};
pub use position::{Position, PositionSide};
pub use trading::TradingStrategy;
