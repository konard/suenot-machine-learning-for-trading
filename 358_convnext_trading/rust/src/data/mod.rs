//! Data module for fetching and processing market data
//!
//! Includes Bybit API integration and feature engineering.

mod bybit;
mod dataset;
mod features;

pub use bybit::{BybitClient, Candle, Interval};
pub use dataset::Dataset;
pub use features::FeatureBuilder;
