//! Machine learning models module
//!
//! This module provides:
//! - Gradient Boosting Machine implementations
//! - Model training, prediction, and evaluation utilities
//! - Cross-validation utilities

pub mod gbm;

pub use gbm::{
    time_series_cv, CrossValidationResult, GbmClassifier, GbmParams, GbmRegressor, ModelError,
    ModelMetrics,
};
