//! # Saliency Maps for Cryptocurrency Trading
//!
//! This library provides implementations of saliency map computation
//! and trading strategies for cryptocurrency markets using Bybit data.
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `data` - Data processing and feature engineering
//! - `models` - Neural network models for price prediction
//! - `saliency` - Saliency map computation methods

pub mod api;
pub mod data;
pub mod models;
pub mod saliency;

pub use api::bybit::BybitClient;
pub use data::features::FeatureEngineering;
pub use data::processor::DataProcessor;
pub use models::network::TradingNetwork;
pub use saliency::gradient::SaliencyComputer;
