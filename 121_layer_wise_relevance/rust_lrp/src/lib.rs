//! Layer-wise Relevance Propagation (LRP) for Trading
//!
//! This crate provides LRP-enabled neural networks for explainable
//! trading predictions with support for multiple propagation rules.
//!
//! # Features
//!
//! - Multiple LRP rules: LRP-0, LRP-epsilon, LRP-gamma, LRP-alpha-beta
//! - Bybit API integration for market data
//! - Technical indicator calculation
//! - Backtesting engine with LRP insights
//!
//! # Example
//!
//! ```rust,ignore
//! use rust_lrp::{
//!     model::{LRPNetwork, LRPConfig},
//!     data::prepare_data,
//!     strategy::backtest,
//! };
//!
//! let config = LRPConfig::default();
//! let model = LRPNetwork::new(420, vec![128, 64, 32], 2, config);
//!
//! let output = model.forward(&input);
//! let relevance = model.explain(&input, Some(1));
//! ```

pub mod api;
pub mod data;
pub mod model;
pub mod strategy;

pub use api::{BybitClient, Kline};
pub use data::{Dataset, Features, prepare_data};
pub use model::{LRPConfig, LRPLinear, LRPNetwork, LRPRule};
pub use strategy::{BacktestConfig, BacktestResults, backtest};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
