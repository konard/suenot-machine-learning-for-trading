//! # Торговые стратегии и бэктестинг
//!
//! Модуль для реализации торговых стратегий на основе предсказаний CNN
//! и их тестирования на исторических данных.

mod backtest;
mod position;
mod signal;
mod strategy;

pub use backtest::Backtest;
pub use position::Position;
pub use signal::Signal;
pub use strategy::Strategy;
