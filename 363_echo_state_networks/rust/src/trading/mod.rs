//! Trading module
//!
//! Feature engineering, signal generation, and backtesting for ESN trading.

mod features;
mod signals;
mod position;
mod backtest;

pub use features::FeatureEngineering;
pub use signals::{TradingSignal, SignalGenerator};
pub use position::{Position, PositionManager};
pub use backtest::{Backtest, BacktestConfig, BacktestResult};
