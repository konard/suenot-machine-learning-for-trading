//! Trading module for signal generation and backtesting.

mod signals;
mod portfolio;
mod backtest;

pub use signals::GraphSignals;
pub use portfolio::{Portfolio, PortfolioOptimizer};
pub use backtest::{BacktestEngine, BacktestResult, Trade};
