//! Backtesting framework module

mod engine;
mod metrics;

pub use engine::{Backtest, BacktestConfig, BacktestResult, Position, Signal};
pub use metrics::PerformanceMetrics;
