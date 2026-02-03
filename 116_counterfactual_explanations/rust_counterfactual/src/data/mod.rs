//! Data loading and feature engineering.

mod features;
mod loader;

pub use features::{compute_macd, compute_rsi, prepare_features};
pub use loader::{create_labels, get_sample_data, Candle};
