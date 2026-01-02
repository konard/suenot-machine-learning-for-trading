//! Trading strategy module based on EBM signals
//!
//! Provides trading signals and position management based on
//! energy-based anomaly detection.

mod signals;
mod position;
mod backtest;

pub use signals::*;
pub use position::*;
pub use backtest::*;
