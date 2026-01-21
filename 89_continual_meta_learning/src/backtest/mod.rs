//! Backtesting framework for CML strategies.
//!
//! This module provides:
//! - Backtest engine
//! - Performance metrics calculation
//! - Trade logging

pub mod engine;

pub use engine::{Backtester, BacktestConfig, BacktestResult, TradeLog};
