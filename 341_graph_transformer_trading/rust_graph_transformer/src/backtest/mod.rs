//! Backtesting framework module
//!
//! Provides backtesting engine and performance reporting for trading strategies.

pub mod engine;
pub mod metrics;
pub mod report;

pub use engine::BacktestEngine;
pub use metrics::PerformanceMetrics;
pub use report::BacktestReport;
