//! # Trading Strategies
//!
//! Strategy implementation and backtesting.

mod rebalancer;
mod backtest;

pub use rebalancer::{ContinuousRebalancer, RebalanceDecision, Trade};
pub use backtest::{BacktestResult, Backtester, BacktestConfig};
