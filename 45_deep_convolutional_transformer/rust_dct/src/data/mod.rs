//! Data processing module
//!
//! Provides data loading, feature engineering, and dataset preparation.

mod dataset;
mod features;
mod loader;

pub use dataset::{DatasetConfig, PreparedDataset, prepare_dataset};
pub use features::{compute_features, create_movement_labels};
pub use loader::OHLCV;
