//! Feature Engineering Module
//!
//! Provides technical indicators and normalization utilities for trading.

mod normalize;
mod technical;

pub use normalize::{NormalizationMethod, Normalizer};
pub use technical::{FeatureMatrix, TechnicalIndicators};
