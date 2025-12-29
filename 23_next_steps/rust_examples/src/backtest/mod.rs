//! Модуль бэктестинга

mod engine;
mod metrics;

pub use engine::{BacktestEngine, BacktestConfig};
pub use metrics::{BacktestResult, PerformanceMetrics};
