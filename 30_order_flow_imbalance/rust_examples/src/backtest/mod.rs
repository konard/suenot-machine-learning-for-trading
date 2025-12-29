//! # Backtesting Module
//!
//! Intraday backtesting framework for order flow strategies.

pub mod engine;
pub mod report;

pub use engine::BacktestEngine;
pub use report::BacktestReport;
