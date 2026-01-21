//! Data handling module for market data from Bybit
//!
//! This module provides:
//! - Bybit API client for fetching market data
//! - Feature extraction for market regime classification
//! - Data types for market data structures

mod bybit;
mod features;
mod types;

pub use bybit::{BybitClient, BybitConfig, BybitError};
pub use features::{FeatureExtractor, MarketFeatures, FeatureConfig};
pub use types::{Kline, OrderBook, OrderBookLevel, Ticker, Trade, FundingRate, OpenInterest, MarketRegime, TradingBias, TradeSide};
