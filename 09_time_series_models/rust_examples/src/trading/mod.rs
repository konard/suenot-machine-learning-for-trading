//! # Торговые стратегии
//!
//! Реализация стратегий на основе временных рядов.

mod cointegration;
mod pairs;
mod signals;
mod backtest;

pub use cointegration::*;
pub use pairs::*;
pub use signals::*;
pub use backtest::*;
