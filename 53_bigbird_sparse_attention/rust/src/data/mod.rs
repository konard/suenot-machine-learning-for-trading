//! Data loading and processing module
//!
//! This module handles data loading, feature engineering, and dataset creation.

mod dataset;
mod features;
mod loader;

pub use dataset::TradingDataset;
pub use features::FeatureEngine;
pub use loader::DataLoader;
