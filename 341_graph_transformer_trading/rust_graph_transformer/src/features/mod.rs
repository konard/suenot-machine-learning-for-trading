//! Feature engineering module
//!
//! Provides feature extraction from market data for Graph Transformer input.

pub mod engine;
pub mod indicators;
pub mod node_features;

pub use engine::FeatureEngine;
pub use indicators::TechnicalIndicators;
pub use node_features::NodeFeatures;
