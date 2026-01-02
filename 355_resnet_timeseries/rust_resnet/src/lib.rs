//! # ResNet for Time Series
//!
//! This crate implements ResNet (Residual Network) architecture adapted for
//! time series analysis in cryptocurrency trading.
//!
//! ## Features
//!
//! - Bybit API client for fetching cryptocurrency data
//! - ResNet model implementation with 1D convolutions
//! - Feature engineering for OHLCV data
//! - Trading strategy with signal generation
//! - Backtesting framework
//!
//! ## Example
//!
//! ```rust,ignore
//! use rust_resnet::{api::BybitClient, model::ResNet18, data::Dataset};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch data
//!     let client = BybitClient::new();
//!     let candles = client.fetch_klines("BTCUSDT", "1", 1000).await?;
//!
//!     // Create dataset
//!     let dataset = Dataset::from_candles(candles, 256, 12)?;
//!
//!     // Create and train model
//!     let model = ResNet18::new(15, 3);
//!     // ... training loop
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod data;
pub mod model;
pub mod strategy;
pub mod utils;

pub use api::BybitClient;
pub use data::{Dataset, Features};
pub use model::{ResNet18, ResidualBlock};
pub use strategy::{TradingSignal, TradingStrategy};
pub use utils::Metrics;
