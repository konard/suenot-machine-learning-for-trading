//! Counterfactual Explanations for Trading
//!
//! This crate provides tools for generating counterfactual explanations
//! for trading models, helping understand what minimal changes would flip
//! a model's prediction.
//!
//! # Example
//!
//! ```no_run
//! use rust_counterfactual::{
//!     model::TradingClassifier,
//!     counterfactual::CounterfactualOptimizer,
//!     data::get_sample_data,
//! };
//!
//! // Load sample data
//! let (features, labels, feature_names) = get_sample_data();
//!
//! // Train a classifier
//! let mut classifier = TradingClassifier::new(features[0].len(), 64, 3);
//! classifier.train(&features, &labels, 100, 0.01);
//!
//! // Generate counterfactual
//! let optimizer = CounterfactualOptimizer::new(&classifier);
//! let result = optimizer.generate(&features[0], 2, 100, 0.01);
//! println!("{}", result);
//! ```

pub mod api;
pub mod counterfactual;
pub mod data;
pub mod model;
pub mod strategy;

pub use api::bybit::BybitClient;
pub use counterfactual::{CounterfactualOptimizer, CounterfactualResult};
pub use data::{prepare_features, Candle};
pub use model::TradingClassifier;
pub use strategy::{Backtester, BacktestResult};
