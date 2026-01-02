//! Feature Engineering Module
//!
//! Extracts technical indicators and features from market data.

mod extractor;
mod indicators;
mod normalizer;

pub use extractor::FeatureExtractor;
pub use indicators::TechnicalIndicators;
pub use normalizer::FeatureNormalizer;
