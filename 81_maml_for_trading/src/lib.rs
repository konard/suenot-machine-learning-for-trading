//! # MAML Meta-Learning for Trading
//!
//! This crate implements the Model-Agnostic Meta-Learning (MAML) algorithm
//! for algorithmic trading. MAML enables rapid adaptation to new market
//! conditions with minimal data.
//!
//! ## Features
//!
//! - Second-order meta-learning (full MAML) and first-order approximation (FOMAML)
//! - Multi-asset training for cross-market generalization
//! - Bybit API integration for cryptocurrency data
//! - Backtesting framework for strategy evaluation
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use maml_trading::{MAMLTrainer, TradingModel, BybitClient};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create model and trainer
//!     let model = TradingModel::new(8, 64, 1);
//!     let trainer = MAMLTrainer::new(model, 0.01, 0.001, 5, false);
//!
//!     // Fetch data and train
//!     let client = BybitClient::new();
//!     let data = client.fetch_klines("BTCUSDT", "60", 1000).await?;
//!
//!     Ok(())
//! }
//! ```

pub mod model;
pub mod maml;
pub mod data;
pub mod trading;
pub mod backtest;

pub use model::network::TradingModel;
pub use maml::algorithm::MAMLTrainer;
pub use data::bybit::BybitClient;
pub use data::features::FeatureGenerator;
pub use trading::strategy::TradingStrategy;
pub use trading::signals::TradingSignal;
pub use backtest::engine::BacktestEngine;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::model::network::TradingModel;
    pub use crate::maml::algorithm::MAMLTrainer;
    pub use crate::data::bybit::BybitClient;
    pub use crate::data::features::FeatureGenerator;
    pub use crate::trading::strategy::TradingStrategy;
    pub use crate::trading::signals::TradingSignal;
    pub use crate::backtest::engine::BacktestEngine;
}

/// Error types for the crate
#[derive(thiserror::Error, Debug)]
pub enum MAMLError {
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
}

pub type Result<T> = std::result::Result<T, MAMLError>;
