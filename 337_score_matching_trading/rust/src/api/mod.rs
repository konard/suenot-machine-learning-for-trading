//! # API Module
//!
//! Provides integration with cryptocurrency exchanges.
//!
//! Currently supports:
//! - Bybit (spot and perpetual markets)

mod bybit;

pub use bybit::{BybitClient, BybitError, KlineResponse, KlineData};
