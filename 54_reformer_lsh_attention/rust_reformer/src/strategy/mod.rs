//! Trading strategy module
//!
//! Provides signal generation, strategy implementation, and backtesting.

mod backtest;
mod signals;

pub use backtest::{BacktestConfig, BacktestResult, run_backtest};
pub use signals::{SignalGenerator, TradingSignal, TradingStrategy};
