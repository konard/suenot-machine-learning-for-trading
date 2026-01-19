//! # Strategy Module
//!
//! Trading signal generation and backtesting for sentiment-based strategies.

mod signals;
mod backtest;

pub use signals::{SignalGenerator, TradingSignal, SignalStrength, PositionSizing};
pub use backtest::{Backtester, BacktestResult, BacktestConfig, Trade};
