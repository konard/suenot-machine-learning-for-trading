//! API module for exchange integrations
//!
//! This module provides clients for fetching market data from cryptocurrency exchanges.

pub mod bybit;

pub use bybit::BybitClient;
