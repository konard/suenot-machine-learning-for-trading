//! Data fetching and processing modules.
//!
//! This module provides clients for fetching market data from:
//! - Bybit (cryptocurrency exchange)
//! - Stock market APIs

pub mod bybit;
pub mod stock;

pub use bybit::{BybitClient, Kline, TickerInfo};
pub use stock::{compute_features, compute_rsi, StockData};
