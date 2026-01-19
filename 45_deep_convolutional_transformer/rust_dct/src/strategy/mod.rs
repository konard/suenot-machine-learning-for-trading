//! Trading Strategy module
//!
//! Provides signal generation and backtesting functionality.

mod backtest;
mod signals;

pub use backtest::{BacktestConfig, BacktestResult, Backtester};
pub use signals::{Signal, SignalGenerator, TradingSignal};
