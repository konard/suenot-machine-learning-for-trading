//! Data module for market data fetching and preprocessing
//!
//! This module provides:
//! - Bybit API client for cryptocurrency data
//! - Feature engineering for GLOW model
//! - Data preprocessing utilities

mod bybit_client;
mod features;
mod preprocessing;

pub use bybit_client::{BybitClient, Candle, Interval, Ticker};
pub use features::{FeatureExtractor, MarketFeatures};
pub use preprocessing::{normalize_features, denormalize_features, Normalizer};
