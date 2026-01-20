//! Factor evaluation module
//!
//! Provides tools for evaluating alpha factors including IC calculation,
//! backtesting, and performance metrics.

mod backtester;
mod ic;
mod metrics;

pub use backtester::{BacktestConfig, BacktestResult, Backtester};
pub use ic::IcCalculator;
pub use metrics::{FactorMetrics, MetricsCalculator, PerformanceMetrics};
