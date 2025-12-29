//! # Bybit API Client
//!
//! Модуль для работы с публичным API биржи Bybit.
//! Поддерживает получение исторических свечей (klines) и текущих рыночных данных.

mod client;
mod types;

pub use client::BybitClient;
pub use types::{Kline, BybitError, BybitResponse, KlineInterval};
