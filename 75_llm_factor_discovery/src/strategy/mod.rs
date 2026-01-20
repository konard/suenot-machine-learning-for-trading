//! Trading strategy module
//!
//! Provides tools for converting factor signals into trading decisions.

mod combiner;
mod execution;
mod signals;

pub use combiner::FactorCombiner;
pub use execution::{ExecutionConfig, TradeAction};
pub use signals::{FactorSignal, FactorStrategy, SignalGenerator, TradingSignal};
