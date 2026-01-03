//! Trading module for GLOW-based strategies
//!
//! This module provides:
//! - GLOWTrader: Trading system using GLOW model
//! - Backtest: Backtesting framework
//! - Performance metrics

mod trader;
mod backtest;
mod metrics;

pub use trader::{GLOWTrader, TraderConfig, TradingSignal};
pub use backtest::{Backtest, BacktestResult, BacktestConfig};
pub use metrics::{PerformanceMetrics, compute_sharpe_ratio, compute_max_drawdown};
