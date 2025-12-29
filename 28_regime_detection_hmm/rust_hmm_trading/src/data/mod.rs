//! Data structures and feature engineering module
//!
//! Provides types for market data and feature engineering for regime detection.

mod types;
mod features;

pub use types::{Candle, Dataset, OHLCV};
pub use features::{build_features, FeatureBuilder, Features};
