//! Rust Gradient Boosting Machine for Cryptocurrency Trading
//!
//! This library provides a complete solution for building gradient boosting
//! based trading strategies using cryptocurrency data from Bybit.
//!
//! # Modules
//!
//! - [`data`] - Market data fetching (Bybit API) and data structures
//! - [`features`] - Technical indicators and feature engineering
//! - [`models`] - Gradient Boosting Machine implementations
//! - [`strategies`] - Trading strategies and backtesting
//!
//! # Example
//!
//! ```rust,no_run
//! use rust_gbm::data::{BybitClient, Interval};
//! use rust_gbm::features::FeatureEngineer;
//! use rust_gbm::models::{GbmRegressor, GbmParams};
//! use rust_gbm::strategies::LongShortStrategy;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // 1. Fetch data from Bybit
//!     let client = BybitClient::new();
//!     let candles = client.get_klines("BTCUSDT", Interval::Hour1, Some(1000), None, None).await?;
//!
//!     // 2. Engineer features
//!     let engineer = FeatureEngineer::new();
//!     let dataset = engineer.build_clean_features(&candles);
//!
//!     // 3. Train model
//!     let (train, test) = dataset.train_test_split(0.8);
//!     let mut model = GbmRegressor::new();
//!     model.fit(&train)?;
//!
//!     // 4. Evaluate
//!     let metrics = model.evaluate(&test)?;
//!     println!("RMSE: {:.4}", metrics.rmse.unwrap_or(0.0));
//!
//!     Ok(())
//! }
//! ```

pub mod data;
pub mod features;
pub mod models;
pub mod strategies;

// Re-export commonly used items at the crate level
pub use data::{BybitClient, Candle, Dataset, Interval};
pub use features::{FeatureConfig, FeatureEngineer};
pub use models::{GbmClassifier, GbmParams, GbmRegressor, ModelError, ModelMetrics};
pub use strategies::{LongShortStrategy, PerformanceMetrics, Signal, StrategyConfig};
