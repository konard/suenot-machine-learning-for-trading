//! # LIME for Cryptocurrency Trading
//!
//! This library provides implementations of LIME (Local Interpretable Model-agnostic
//! Explanations) for explaining trading model predictions using cryptocurrency data
//! from Bybit exchange.
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `data` - Data processing and feature engineering
//! - `explainer` - Core LIME algorithm implementation
//! - `models` - Trading model implementations and wrappers

pub mod api;
pub mod data;
pub mod explainer;
pub mod models;

pub use api::bybit::BybitClient;
pub use data::features::FeatureEngineering;
pub use data::processor::DataProcessor;
pub use explainer::lime::{LimeExplainer, LimeExplanation};
pub use models::random_forest::RandomForestClassifier;
