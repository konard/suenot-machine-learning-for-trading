//! Data loading and processing utilities.

mod bybit;
mod features;

pub use bybit::{generate_synthetic_candles, BybitClient, Candle};
pub use features::{compute_returns, compute_technical_features, FeatureSet};
