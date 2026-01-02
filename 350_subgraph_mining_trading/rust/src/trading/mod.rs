//! Trading module for signal generation and strategy execution.
//!
//! This module provides:
//! - Signal generation from pattern matches
//! - Trading strategy implementation
//! - Backtesting framework

mod signals;
mod strategy;
mod backtest;

pub use signals::{Signal, SignalType, SignalGenerator};
pub use strategy::{Strategy, StrategyConfig, Position};
pub use backtest::{Backtester, BacktestResult, BacktestConfig, Trade};
