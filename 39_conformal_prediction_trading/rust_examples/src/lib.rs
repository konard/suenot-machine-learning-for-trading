//! # Conformal Prediction for Cryptocurrency Trading
//!
//! This library provides implementations of conformal prediction methods
//! for cryptocurrency trading using data from Bybit exchange.
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `data` - Data processing and feature engineering
//! - `conformal` - Conformal prediction algorithms (Split CP, CQR, ACI)
//! - `strategy` - Trading strategies with uncertainty-based signals
//! - `metrics` - Evaluation metrics for coverage and trading performance
//!
//! ## Example
//!
//! ```rust,no_run
//! use conformal_prediction_trading::{
//!     api::bybit::{BybitClient, Interval},
//!     data::features::FeatureEngineering,
//!     conformal::split::SplitConformalPredictor,
//!     strategy::trading::ConformalTradingStrategy,
//! };
//!
//! fn main() -> anyhow::Result<()> {
//!     // Fetch data from Bybit
//!     let client = BybitClient::new();
//!     let klines = client.get_klines("BTCUSDT", Interval::Hour1, Some(500), None, None)?;
//!
//!     // Generate features and targets
//!     let features = FeatureEngineering::generate_features(&klines);
//!     let targets = FeatureEngineering::create_returns(&klines, 1);
//!
//!     // Train conformal predictor
//!     // ... (see examples for complete workflow)
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod conformal;
pub mod data;
pub mod metrics;
pub mod strategy;

// Re-exports for convenience
pub use api::bybit::BybitClient;
pub use conformal::adaptive::AdaptiveConformalPredictor;
pub use conformal::split::SplitConformalPredictor;
pub use data::features::FeatureEngineering;
pub use data::processor::DataProcessor;
pub use metrics::coverage::CoverageMetrics;
pub use metrics::trading::TradingMetrics;
pub use strategy::sizing::PositionSizer;
pub use strategy::trading::ConformalTradingStrategy;
