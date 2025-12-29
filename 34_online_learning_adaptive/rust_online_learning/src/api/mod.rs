//! Bybit API Client Module
//!
//! This module provides functionality to fetch cryptocurrency market data
//! from the Bybit exchange.

mod client;
mod error;

pub use client::{BybitClient, Candle, Interval};
pub use error::{ApiError, ApiResult};
