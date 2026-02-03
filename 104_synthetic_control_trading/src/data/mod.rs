//! Data module for fetching market data.
//!
//! Supports data from Bybit and synthetic data generation.

pub mod bybit;

pub use bybit::BybitClient;
