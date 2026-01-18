//! Data loading and feature engineering module
//!
//! This module provides functionality for:
//! - Loading and processing market data
//! - Computing technical indicators
//! - Preparing data for TABL model

mod features;
mod loader;

pub use features::{prepare_features, Features};
pub use loader::DataLoader;
