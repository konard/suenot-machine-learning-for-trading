//! # Bybit API Module
//!
//! Модуль для работы с API биржи Bybit.
//! Поддерживает получение OHLCV данных, стакана заявок и тикеров.

mod client;
mod types;

pub use client::BybitClient;
pub use types::{BybitError, Kline, OrderBook, OrderBookLevel, Ticker};
