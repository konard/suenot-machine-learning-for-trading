//! Trading strategy module for FNet.

pub mod backtest;
pub mod signals;

pub use backtest::{BacktestResult, Backtester, TradeMetrics};
pub use signals::{Signal, SignalGenerator, TradingSignal};
