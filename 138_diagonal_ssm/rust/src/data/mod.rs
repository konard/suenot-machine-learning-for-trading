//! Data loading, feature engineering, and dataset construction.
//!
//! This module provides tools to transform raw kline data into
//! feature matrices suitable for training the Diagonal SSM model.

pub mod dataset;
pub mod features;
pub mod loader;
