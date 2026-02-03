//! # Prototype Learning for Trading
//!
//! This library provides implementations of prototype-based learning models
//! for market state classification and trading signal generation.
//!
//! ## Features
//!
//! - Bybit API client for fetching cryptocurrency market data
//! - Feature engineering for technical indicators
//! - Prototype network inference for market state classification
//! - Trading metrics and performance evaluation
//!
//! ## Example
//!
//! ```rust,no_run
//! use prototype_learning::{BybitClient, Interval, FeatureEngineering, PrototypeNetwork};
//!
//! // Fetch data
//! let client = BybitClient::new();
//! let klines = client.get_klines("BTCUSDT", Interval::Hour1, Some(1000), None, None)?;
//!
//! // Compute features
//! let fe = FeatureEngineering::new(60);
//! let features = fe.compute_features(&klines)?;
//!
//! // Classify market state
//! let model = PrototypeNetwork::new(13, 128, 5, 2);
//! let (predictions, confidences) = model.predict(&features)?;
//! ```

pub mod api;
pub mod data;
pub mod metrics;
pub mod models;

pub use api::bybit::{BybitClient, BybitError, Interval, Kline, TickerInfo};
pub use data::features::FeatureEngineering;
pub use data::processor::DataProcessor;
pub use metrics::classification::ClassificationMetrics;
pub use metrics::trading::TradingMetrics;
pub use models::prototype::{MarketState, PrototypeNetwork};
