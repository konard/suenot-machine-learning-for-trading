//! Data module for fetching and processing cryptocurrency market data.
//!
//! This module provides:
//! - Bybit API client for fetching OHLCV data
//! - Returns calculation utilities
//! - Data structures for market data

pub mod bybit;
pub mod returns;

pub use bybit::{BybitClient, Kline, Symbol};
pub use returns::Returns;
