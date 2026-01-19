//! Data processing module for FNet trading.

pub mod dataset;
pub mod features;

pub use dataset::TradingDataset;
pub use features::{calculate_features, FeatureConfig, TradingFeatures};
