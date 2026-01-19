//! Trading strategy module
//!
//! Backtesting and signal generation for trading strategies.

mod backtest;
mod signals;

pub use backtest::{BacktestConfig, BacktestResult, Backtester};
pub use signals::{Signal, SignalGenerator, SignalType};
