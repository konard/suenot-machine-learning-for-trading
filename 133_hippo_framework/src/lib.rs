//! # HiPPO Framework for Trading
//!
//! This crate implements the HiPPO (High-order Polynomial Projection Operators)
//! framework for algorithmic trading. HiPPO provides optimal polynomial projections
//! for compressing continuous signals into fixed-dimensional state representations,
//! enabling long-range memory in time series modeling.
//!
//! ## Features
//!
//! - HiPPO-LegS (Scaled Legendre) and HiPPO-LagT (Translated Laguerre) operators
//! - Bilinear (Tustin) discretization for numerical stability
//! - Bybit API integration for cryptocurrency data
//! - Trading strategy based on polynomial coefficient dynamics
//! - Backtesting framework for strategy evaluation
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use hippo_trading::{HiPPOLegS, TradingStrategy, BybitClient};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let hippo = HiPPOLegS::new(64);
//!     let client = BybitClient::new();
//!     let data = client.fetch_klines("BTCUSDT", "60", 1000).await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Reference
//!
//! Gu et al., "HiPPO: Recurrent Memory with Optimal Polynomial Projections"
//! NeurIPS 2020. arXiv:2008.07669

pub mod model;
pub mod data;
pub mod trading;
pub mod backtest;

pub use model::hippo::{HiPPOLegS, HiPPOLagT};
pub use data::bybit::BybitClient;
pub use trading::strategy::TradingStrategy;
pub use trading::signals::TradingSignal;
pub use backtest::engine::BacktestEngine;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::model::hippo::{HiPPOLegS, HiPPOLagT};
    pub use crate::data::bybit::BybitClient;
    pub use crate::trading::strategy::TradingStrategy;
    pub use crate::trading::signals::TradingSignal;
    pub use crate::backtest::engine::BacktestEngine;
}

/// Error types for the crate.
#[derive(thiserror::Error, Debug)]
pub enum HiPPOError {
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
}

pub type Result<T> = std::result::Result<T, HiPPOError>;
