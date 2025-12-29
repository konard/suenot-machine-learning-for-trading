//! Data Module
//!
//! Provides data fetching and preprocessing capabilities:
//! - Bybit API client for fetching cryptocurrency data
//! - OHLCV data structures
//! - Data normalization utilities

mod bybit;
mod ohlcv;
mod normalize;

pub use bybit::BybitClient;
pub use ohlcv::{OHLCV, OHLCVSeries};
pub use normalize::{Normalizer, MinMaxNormalizer, StandardNormalizer};
