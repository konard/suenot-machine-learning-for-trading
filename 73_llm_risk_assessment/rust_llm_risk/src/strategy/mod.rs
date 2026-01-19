//! Trading strategy module.

mod trader;
mod signal;
mod backtest;

pub use trader::RiskBasedTrader;
pub use signal::{TradingSignal, SignalType, Position};
pub use backtest::{BacktestConfig, BacktestResult, Backtester};
