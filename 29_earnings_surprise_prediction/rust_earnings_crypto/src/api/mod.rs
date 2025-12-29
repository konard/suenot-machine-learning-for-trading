//! API module for exchange interactions
//!
//! This module provides clients for fetching market data from cryptocurrency exchanges.

mod bybit;
mod error;

pub use bybit::{BybitClient, Interval};
pub use error::{ApiError, ApiResult};
