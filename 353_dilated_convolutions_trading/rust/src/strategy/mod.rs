//! Trading Strategy Module
//!
//! This module implements trading strategies using dilated convolutions.

pub mod position;
pub mod signals;

pub use position::{Position, PositionSizer};
pub use signals::{Signal, TradingStrategy};
