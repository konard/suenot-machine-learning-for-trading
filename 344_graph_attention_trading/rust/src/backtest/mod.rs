//! Backtesting framework
//!
//! Evaluate trading strategies on historical data.

mod metrics;

pub use metrics::{Backtester, BacktestResult, PerformanceMetrics};
