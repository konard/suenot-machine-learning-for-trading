//! Data loading and processing module.
//!
//! This module provides components for fetching and processing market data:
//! - Bybit API integration for cryptocurrency data
//! - Feature engineering for trading signals
//! - Market regime detection

pub mod bybit;
pub mod features;

pub use bybit::{BybitClient, Kline, BybitError};
pub use features::{TradingFeatures, FeatureConfig};
