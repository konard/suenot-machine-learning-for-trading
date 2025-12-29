//! # Crypto ML - Machine Learning for Crypto Trading
//!
//! This library provides implementations of Decision Trees and Random Forests
//! for cryptocurrency trading using Bybit market data.
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `data` - Data structures and preprocessing
//! - `features` - Feature engineering for ML models
//! - `models` - Decision Tree and Random Forest implementations
//! - `backtest` - Backtesting framework for trading strategies

pub mod api;
pub mod backtest;
pub mod data;
pub mod features;
pub mod models;

pub use api::BybitClient;
pub use data::{Candle, Dataset};
pub use features::FeatureEngine;
pub use models::{DecisionTree, RandomForest};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::api::BybitClient;
    pub use crate::backtest::{Backtest, BacktestResult, Position, Signal};
    pub use crate::data::{Candle, Dataset, Split};
    pub use crate::features::{Feature, FeatureEngine};
    pub use crate::models::{DecisionTree, RandomForest, TreeConfig};
}
