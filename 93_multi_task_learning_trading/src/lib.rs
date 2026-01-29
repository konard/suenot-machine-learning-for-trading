//! # Multi-Task Learning for Trading
//!
//! This crate implements Multi-Task Learning (MTL) for algorithmic trading.
//! MTL trains a single model to solve multiple related tasks simultaneously
//! (return prediction, volatility estimation, direction classification,
//! volume forecasting), sharing representations to improve generalization.
//!
//! ## Features
//!
//! - Shared-encoder architecture with task-specific heads
//! - Uncertainty-based task weighting (Kendall et al., 2018)
//! - Bybit API integration for cryptocurrency data
//! - Feature engineering pipeline
//! - Backtesting framework for strategy evaluation
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use mtl_trading::{MultiTaskModel, MTLTrainer, BybitClient};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let model = MultiTaskModel::new(10, &[64, 32], 16);
//!     let trainer = MTLTrainer::new(model, 0.001, true);
//!
//!     let client = BybitClient::new();
//!     let data = client.fetch_klines("BTCUSDT", "60", 1000).await?;
//!
//!     Ok(())
//! }
//! ```

pub mod model;
pub mod mtl;
pub mod data;
pub mod trading;
pub mod backtest;

pub use model::network::MultiTaskModel;
pub use mtl::trainer::MTLTrainer;
pub use data::bybit::BybitClient;
pub use data::features::FeatureGenerator;
pub use trading::strategy::TradingStrategy;
pub use trading::signals::TradingSignal;
pub use backtest::engine::BacktestEngine;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::model::network::MultiTaskModel;
    pub use crate::mtl::trainer::MTLTrainer;
    pub use crate::data::bybit::BybitClient;
    pub use crate::data::features::FeatureGenerator;
    pub use crate::trading::strategy::TradingStrategy;
    pub use crate::trading::signals::TradingSignal;
    pub use crate::backtest::engine::BacktestEngine;
}

/// Error types for the crate
#[derive(thiserror::Error, Debug)]
pub enum MTLError {
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

    #[error("Training error: {0}")]
    TrainingError(String),
}

pub type Result<T> = std::result::Result<T, MTLError>;
