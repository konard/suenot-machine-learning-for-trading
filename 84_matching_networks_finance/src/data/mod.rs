//! Data fetching and preprocessing module
//!
//! Provides utilities for:
//! - Fetching cryptocurrency data from Bybit API
//! - Fetching stock market data
//! - Feature extraction for market analysis
//! - Market pattern definitions

mod pattern;
mod features;
#[cfg(feature = "bybit")]
mod bybit;
mod stock;

pub use pattern::MarketPattern;
pub use features::{MarketFeatures, FeatureExtractor, OHLCVBar};
#[cfg(feature = "bybit")]
pub use bybit::{BybitClient, BybitConfig, KlineInterval};
pub use stock::StockDataLoader;
