//! Trading strategy and backtesting module.
//!
//! Provides backtesting framework and performance metrics calculation.

pub mod backtest;
pub mod metrics;

pub use backtest::{BacktestConfig, BacktestResult, Backtester};
pub use metrics::PerformanceMetrics;
