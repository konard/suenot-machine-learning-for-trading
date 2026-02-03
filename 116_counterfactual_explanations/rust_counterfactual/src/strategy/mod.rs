//! Trading strategy and backtesting.

mod backtest;
mod signals;

pub use backtest::{Backtester, BacktestResult, Trade};
pub use signals::generate_signals;
