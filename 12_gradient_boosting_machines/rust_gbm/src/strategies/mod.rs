//! Trading strategies module
//!
//! This module provides:
//! - Long-short trading strategy based on GBM predictions
//! - Backtesting utilities
//! - Performance metrics calculation

pub mod long_short;

pub use long_short::{
    print_backtest_summary, LongShortStrategy, PerformanceMetrics, Position, Signal,
    StrategyConfig, Trade,
};
