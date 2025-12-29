//! Feature engineering module
//!
//! Provides technical indicators and feature generation for ML models.

mod engine;
mod indicators;

pub use engine::{Feature, FeatureEngine};
pub use indicators::*;
