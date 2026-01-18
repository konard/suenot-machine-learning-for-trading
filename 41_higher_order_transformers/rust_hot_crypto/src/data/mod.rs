//! Data module for fetching and processing market data
//!
//! This module provides:
//! - Bybit API client for fetching OHLCV data
//! - Data types for candles and price series
//! - Feature engineering utilities

mod bybit;
mod types;

pub use bybit::BybitClient;
pub use types::{Candle, PriceSeries, Features};
