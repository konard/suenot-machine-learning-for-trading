//! Data module: market data fetching and feature engineering.

pub mod bybit;
pub mod features;

pub use bybit::{BybitClient, Candle};
pub use features::{compute_features, FeatureRow};
