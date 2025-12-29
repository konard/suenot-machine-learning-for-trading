//! Модуль для работы с данными
//!
//! Этот модуль содержит:
//! - Типы данных (Candle, PriceSeries, Portfolio, Signal)
//! - Клиент для Bybit API
//! - Функции для загрузки и обработки данных

pub mod bybit;
pub mod types;

pub use bybit::{get_momentum_universe, BybitClient, CRYPTO_UNIVERSE};
pub use types::{Candle, Portfolio, PriceSeries, Signal, Signals};
