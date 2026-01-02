//! Trading Module
//!
//! Provides signal generation, risk management, and backtesting capabilities.

mod backtest;
mod risk;
mod signal;

pub use backtest::{BacktestConfig, BacktestEngine, BacktestResult, Trade};
pub use risk::{RiskConfig, RiskManager, ValidatedSignal};
pub use signal::{SignalGenerator, SignalType, TradingSignal};
