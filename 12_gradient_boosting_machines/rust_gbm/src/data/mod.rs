//! Data module for fetching and processing market data
//!
//! This module provides:
//! - Bybit API client for fetching cryptocurrency data
//! - Data types for candles, order books, and trades
//! - Dataset structures for machine learning

pub mod bybit;
pub mod types;

pub use bybit::BybitClient;
pub use types::{Candle, Dataset, Interval, OrderBook, Trade, TradeSide};
