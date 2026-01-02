//! Backtesting module
//!
//! This module provides:
//! - Backtesting engine
//! - Performance analytics

mod analytics;
mod engine;

pub use analytics::{BacktestResult, PerformanceMetrics};
pub use engine::BacktestEngine;
