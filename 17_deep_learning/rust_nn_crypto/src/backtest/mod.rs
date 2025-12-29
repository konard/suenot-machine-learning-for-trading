//! Backtesting Module
//!
//! Provides backtesting engine and performance metrics

mod engine;
mod metrics;

pub use engine::Backtester;
pub use metrics::BacktestMetrics;
