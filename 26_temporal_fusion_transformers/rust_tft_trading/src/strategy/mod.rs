//! # Strategy Module
//!
//! Торговые стратегии и бэктестинг на основе прогнозов TFT.

mod backtest;
mod signals;

pub use backtest::{BacktestConfig, BacktestResult, Backtester};
pub use signals::{Signal, SignalGenerator, TradingStrategy, Position};
