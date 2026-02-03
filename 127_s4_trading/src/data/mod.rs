//! Data loading utilities for S4 trading.
//!
//! Provides data loaders for:
//! - Bybit API (cryptocurrency data)
//! - Synthetic data generation (testing)

pub mod bybit;

pub use bybit::BybitClient;
