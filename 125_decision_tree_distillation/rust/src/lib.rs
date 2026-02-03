//! # Decision Tree Distillation for Trading
//!
//! This crate provides tools for distilling complex machine learning models
//! into interpretable decision trees for algorithmic trading applications.
//!
//! ## Features
//!
//! - Decision tree implementation optimized for distillation
//! - Rule extraction from trained trees
//! - Bybit API integration for cryptocurrency data
//! - Backtesting framework for comparing distilled models
//!
//! ## Example
//!
//! ```rust,no_run
//! use decision_tree_distillation::{DecisionTree, DistillationConfig};
//!
//! let config = DistillationConfig {
//!     max_depth: 5,
//!     min_samples_leaf: 50,
//!     min_samples_split: 100,
//! };
//!
//! let mut tree = DecisionTree::new(config);
//! // tree.fit(&features, &soft_labels);
//! // let rules = tree.extract_rules();
//! ```

pub mod model;
pub mod data;
pub mod backtest;
pub mod trading;

pub use model::{DecisionTree, DistillationConfig, DistilledRule, TreeNode};
pub use data::{BybitClient, MarketData, TechnicalFeatures};
pub use backtest::{Backtester, BacktestResult, Trade};
pub use trading::{TradingSignal, Position};

/// Result type for distillation operations
pub type Result<T> = std::result::Result<T, DistillationError>;

/// Errors that can occur during distillation
#[derive(Debug, thiserror::Error)]
pub enum DistillationError {
    #[error("Model not fitted")]
    NotFitted,

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Data error: {0}")]
    DataError(String),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}
