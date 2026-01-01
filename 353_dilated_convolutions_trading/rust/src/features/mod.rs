//! Feature Engineering Module
//!
//! This module provides technical indicators and feature extraction
//! for trading models.

pub mod normalization;
pub mod technical;

pub use normalization::Normalizer;
pub use technical::TechnicalFeatures;
