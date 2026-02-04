//! Feature engineering module
//!
//! Provides feature computation from OHLCV data.

pub mod engine;
pub mod indicators;

pub use engine::FeatureEngine;
pub use indicators::TechnicalIndicators;
