//! Data processing and feature engineering modules.
//!
//! - `features` - Technical indicators and feature generation
//! - `processor` - Data preprocessing and splitting utilities

pub mod features;
pub mod processor;

pub use features::FeatureEngineering;
pub use processor::DataProcessor;
