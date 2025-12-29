//! Feature engineering module
//!
//! Provides tools for calculating trading features:
//! - Technical indicators
//! - Volume analysis
//! - Surprise metrics

mod technical;
mod volume;
mod surprise;

pub use technical::TechnicalIndicators;
pub use volume::VolumeAnalyzer;
pub use surprise::{SurpriseCalculator, SurpriseMetrics};
