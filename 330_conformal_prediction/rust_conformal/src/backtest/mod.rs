//! Backtesting module
//!
//! Provides backtesting engine and performance metrics.

pub mod engine;
pub mod metrics;

pub use engine::BacktestEngine;
pub use metrics::PerformanceMetrics;
