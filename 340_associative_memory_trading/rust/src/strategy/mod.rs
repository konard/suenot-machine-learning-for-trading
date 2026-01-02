//! Trading strategy module
//!
//! Provides:
//! - Signal generation based on associative memory
//! - Position management
//! - Backtesting framework

pub mod signals;
pub mod backtest;

pub use signals::*;
pub use backtest::*;
