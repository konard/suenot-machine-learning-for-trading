//! Trading strategy module

mod signals;
mod backtest;

pub use signals::{Signal, SignalGenerator, analyze_signal};
pub use backtest::{BacktestConfig, BacktestResults, backtest};
