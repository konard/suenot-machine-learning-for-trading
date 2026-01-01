//! Data handling and API integration
//!
//! This module provides functionality for fetching market data from Bybit
//! and computing technical indicators for the SE trading model.

pub mod bybit;
pub mod features;
pub mod normalize;

pub use bybit::{BybitClient, Kline, OrderBook};
pub use features::FeatureEngine;
pub use normalize::Normalizer;
