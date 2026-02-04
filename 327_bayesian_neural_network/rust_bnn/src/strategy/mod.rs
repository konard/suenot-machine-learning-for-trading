//! Trading strategy module.

pub mod signal;
pub mod trading;

pub use signal::{Signal, SignalGenerator, SignalType};
pub use trading::TradingStrategy;
