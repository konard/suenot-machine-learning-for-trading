//! Feature engineering module
//!
//! This module provides:
//! - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
//! - Feature engineering for ML models

pub mod engineering;
pub mod technical;

pub use engineering::{FeatureConfig, FeatureEngineer};
