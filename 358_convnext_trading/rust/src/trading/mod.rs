//! Trading module for signals, strategy, and backtesting

mod backtest;
mod signals;
mod strategy;

pub use backtest::{Backtest, BacktestMetrics};
pub use signals::Signal;
pub use strategy::Strategy;
