//! # ML Crypto - Machine Learning for Cryptocurrency Trading
//!
//! This library provides tools for machine learning with cryptocurrency data,
//! focusing on Bybit exchange. It demonstrates the ML workflow concepts:
//!
//! - Data fetching from Bybit API
//! - Feature engineering for trading
//! - ML algorithms (KNN, etc.)
//! - Cross-validation and model selection
//! - Bias-variance tradeoff analysis

pub mod api;
pub mod data;
pub mod features;
pub mod ml;

pub use api::bybit::BybitClient;
pub use data::types::{Candle, OrderBook, Trade};
pub use features::engineering::FeatureEngine;
pub use ml::knn::KNNClassifier;
pub use ml::cross_validation::CrossValidator;
pub use ml::metrics::Metrics;
