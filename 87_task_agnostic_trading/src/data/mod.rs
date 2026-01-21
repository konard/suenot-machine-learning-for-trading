//! Data handling and Bybit API integration
//!
//! This module provides:
//! - Bybit API client for fetching market data
//! - Feature extraction from raw market data
//! - Data types for market information

mod bybit;
mod features;
mod types;

pub use bybit::{BybitClient, BybitConfig};
pub use features::{FeatureExtractor, FeatureConfig, MarketFeatures};
pub use types::{Kline, Trade, OrderBook, FundingRate, Ticker};
