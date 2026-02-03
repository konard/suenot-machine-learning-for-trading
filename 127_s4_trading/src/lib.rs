//! # S4 Trading - Structured State Space Models for Financial Markets
//!
//! This crate implements S4 (Structured State Space sequence models) for trading
//! signal generation and price prediction. S4 efficiently handles long-range
//! dependencies in financial time series with linear computational complexity.
//!
//! ## Features
//!
//! - S4 layer implementation with HiPPO initialization
//! - Diagonal S4 (S4D) for simplified deployment
//! - Bybit API integration for cryptocurrency data
//! - Trading strategy with S4-based signal generation
//! - Backtesting framework for strategy evaluation
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use s4_trading::{S4Model, TradingStrategy, BybitClient};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let client = BybitClient::new();
//!     let data = client.fetch_klines("BTCUSDT", "60", 1000).await?;
//!
//!     let model = S4Model::new(S4Config::default());
//!     let signal = model.predict(&features);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## References
//!
//! Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces"
//! ICLR 2022. arXiv:2111.00396

pub mod model;
pub mod data;
pub mod trading;
pub mod backtest;

pub use model::s4::{S4Layer, S4Model, S4Config, S4DLayer};
pub use data::bybit::BybitClient;
pub use trading::strategy::TradingStrategy;
pub use trading::signals::TradingSignal;
pub use backtest::engine::BacktestEngine;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::model::s4::{S4Layer, S4Model, S4Config, S4DLayer};
    pub use crate::data::bybit::BybitClient;
    pub use crate::trading::strategy::TradingStrategy;
    pub use crate::trading::signals::TradingSignal;
    pub use crate::backtest::engine::BacktestEngine;
}

/// Error types for the crate.
#[derive(thiserror::Error, Debug)]
pub enum S4Error {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("JSON parsing failed: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Insufficient data: need {needed}, got {got}")]
    InsufficientData { needed: usize, got: usize },

    #[error("Computation error: {0}")]
    ComputationError(String),

    #[error("Model not initialized")]
    ModelNotInitialized,
}

pub type Result<T> = std::result::Result<T, S4Error>;
