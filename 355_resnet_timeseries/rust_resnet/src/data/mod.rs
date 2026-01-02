//! Data processing module for time series
//!
//! This module handles feature engineering, dataset creation, and preprocessing.

mod dataset;
mod features;
mod preprocessing;

pub use dataset::{Dataset, Sample};
pub use features::Features;
pub use preprocessing::{Normalizer, StandardScaler};
