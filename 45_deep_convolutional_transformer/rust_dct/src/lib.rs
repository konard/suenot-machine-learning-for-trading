//! # Deep Convolutional Transformer (DCT) for Stock Movement Prediction
//!
//! A high-performance Rust implementation of the DCT architecture for
//! predicting stock price movement direction.
//!
//! ## Features
//!
//! - Bybit API integration for crypto data
//! - Inception-style convolutional embedding
//! - Multi-head self-attention
//! - Backtesting with realistic transaction costs
//!
//! ## Example
//!
//! ```rust,no_run
//! use rust_dct::{api::BybitClient, model::DCTConfig};
//!
//! #[tokio::main]
//! async fn main() {
//!     let client = BybitClient::new();
//!     let config = DCTConfig::default();
//!     // ... use client and config ...
//! }
//! ```

pub mod api;
pub mod data;
pub mod model;
pub mod strategy;

pub use api::BybitClient;
pub use data::{DatasetConfig, PreparedDataset, OHLCV, prepare_dataset};
pub use model::{DCTConfig, DCTModel};
pub use strategy::{BacktestConfig, BacktestResult, Backtester};
