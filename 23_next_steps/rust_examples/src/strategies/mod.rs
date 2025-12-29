//! Торговые стратегии

mod base;
mod sma_cross;
mod rsi_oversold;

pub use base::{Signal, Strategy, Position, Trade};
pub use sma_cross::SmaCrossStrategy;
pub use rsi_oversold::RsiStrategy;
