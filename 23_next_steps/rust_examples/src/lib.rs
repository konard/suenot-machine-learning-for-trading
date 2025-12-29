//! # ML4T Bybit
//!
//! Библиотека для работы с криптовалютной биржей Bybit
//! в контексте машинного обучения для трейдинга.
//!
//! ## Модули
//!
//! - `client` - Клиент для работы с Bybit API
//! - `data` - Структуры данных для рыночной информации
//! - `indicators` - Технические индикаторы
//! - `strategies` - Торговые стратегии
//! - `backtest` - Движок бэктестинга

pub mod client;
pub mod data;
pub mod indicators;
pub mod strategies;
pub mod backtest;

pub use client::BybitClient;
pub use data::{Kline, Interval};
pub use indicators::{SMA, EMA, RSI, MACD, BollingerBands};
pub use strategies::{Strategy, Signal, SmaCrossStrategy, RsiStrategy};
pub use backtest::{BacktestEngine, BacktestConfig, BacktestResult};
