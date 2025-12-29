//! # API Module
//!
//! Клиент для работы с Bybit API.

mod client;
mod types;
mod error;

pub use client::BybitClient;
pub use types::{Candle, OrderBook, OrderBookLevel, Trade, Interval};
pub use error::ApiError;
