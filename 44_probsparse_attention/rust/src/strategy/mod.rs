//! Торговая стратегия и бэктестинг
//!
//! - `signals` - Генератор торговых сигналов
//! - `backtest` - Бэктестинг стратегии

pub mod signals;
pub mod backtest;

pub use signals::{TradingSignal, SignalGenerator};
pub use backtest::{BacktestResult, TradingStrategy};
