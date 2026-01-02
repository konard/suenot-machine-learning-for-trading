//! Backtesting engine module

mod engine;
mod metrics;

pub use engine::{BacktestEngine, BacktestConfig, BacktestResult};
pub use metrics::{PerformanceMetrics, TradeStats};
