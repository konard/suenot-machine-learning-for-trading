//! # Meta-Gradient Optimization for Trading
//!
//! This crate implements Meta-Gradient Optimization for algorithmic trading.
//! Meta-gradient optimization learns the hyperparameters of the learning process
//! (learning rates, loss function parameters) through gradient-based methods,
//! enabling self-tuning trading strategies.
//!
//! ## Features
//!
//! - Learnable per-parameter learning rates via log-space parameterization
//! - Learnable loss function with directional and asymmetric components
//! - Online meta-gradient adaptation for live trading
//! - Bybit API integration for cryptocurrency data
//! - Backtesting framework for strategy evaluation
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use meta_gradient_trading::{MetaGradientOptimizer, TradingModel, BybitClient};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let model = TradingModel::new(8, 64, 1);
//!     let optimizer = MetaGradientOptimizer::new(model, 0.01, 0.001, 5, true, false);
//!
//!     let client = BybitClient::new();
//!     let data = client.fetch_klines("BTCUSDT", "60", 1000).await?;
//!
//!     Ok(())
//! }
//! ```

pub mod model;
pub mod meta_gradient;
pub mod data;
pub mod trading;
pub mod backtest;

pub use model::network::TradingModel;
pub use meta_gradient::optimizer::MetaGradientOptimizer;
pub use data::bybit::BybitClient;
pub use data::features::FeatureGenerator;
pub use trading::strategy::TradingStrategy;
pub use trading::signals::TradingSignal;
pub use backtest::engine::BacktestEngine;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::model::network::TradingModel;
    pub use crate::meta_gradient::optimizer::MetaGradientOptimizer;
    pub use crate::data::bybit::BybitClient;
    pub use crate::data::features::FeatureGenerator;
    pub use crate::trading::strategy::TradingStrategy;
    pub use crate::trading::signals::TradingSignal;
    pub use crate::backtest::engine::BacktestEngine;
}

/// Error types for the crate
#[derive(thiserror::Error, Debug)]
pub enum MetaGradientError {
    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Data error: {0}")]
    DataError(String),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Backtest error: {0}")]
    BacktestError(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Gradient computation error: {0}")]
    GradientError(String),

    #[error("Optimization error: {0}")]
    OptimizationError(String),
}

pub type Result<T> = std::result::Result<T, MetaGradientError>;
