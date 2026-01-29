//! Backtesting engine for task-agnostic trading strategies
//!
//! This module provides:
//! - Backtesting engine that simulates trading over historical data
//! - Performance metrics (Sharpe, Sortino, max drawdown, etc.)

mod engine;

pub use engine::{BacktestEngine, BacktestConfig, BacktestResult, PerformanceMetrics};
