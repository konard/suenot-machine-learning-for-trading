//! # Linear Models for Cryptocurrency Trading
//!
//! This library provides implementations of linear models for predicting
//! cryptocurrency prices using data from Bybit exchange.
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `data` - Data processing and feature engineering
//! - `models` - Linear regression, logistic regression, Ridge, Lasso
//! - `metrics` - Model evaluation metrics

pub mod api;
pub mod data;
pub mod metrics;
pub mod models;

pub use api::bybit::BybitClient;
pub use data::features::FeatureEngineering;
pub use data::processor::DataProcessor;
pub use metrics::regression::RegressionMetrics;
pub use metrics::classification::ClassificationMetrics;
pub use models::linear::LinearRegression;
pub use models::logistic::LogisticRegression;
pub use models::regularization::{LassoRegression, RidgeRegression};
