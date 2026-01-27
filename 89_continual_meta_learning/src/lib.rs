//! # Continual Meta-Learning for Trading
//!
//! This crate implements Continual Meta-Learning for algorithmic trading,
//! combining MAML (Model-Agnostic Meta-Learning) with continual learning
//! techniques (EWC + experience replay) to learn new market regimes
//! without forgetting previously learned ones.
//!
//! ## Key Features
//!
//! - MAML/FOMAML for fast adaptation to new market conditions
//! - Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting
//! - Experience replay buffer for revisiting past market regimes
//! - Bybit API integration for cryptocurrency data
//! - Backtesting framework for strategy evaluation
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use continual_meta_learning::{ContinualMAMLTrainer, TradingModel, BybitClient};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let model = TradingModel::new(11, 64, 1);
//!     let trainer = ContinualMAMLTrainer::new(model, 0.01, 0.001, 5, true, 100.0, 50);
//!
//!     let client = BybitClient::new();
//!     let data = client.fetch_klines("BTCUSDT", "60", 1000).await?;
//!
//!     Ok(())
//! }
//! ```

pub mod model;
pub mod continual;
pub mod data;
pub mod trading;
pub mod backtest;

pub use model::network::TradingModel;
pub use continual::algorithm::ContinualMAMLTrainer;
pub use data::bybit::BybitClient;
pub use data::features::FeatureGenerator;
pub use trading::strategy::TradingStrategy;
pub use trading::signals::TradingSignal;
pub use backtest::engine::BacktestEngine;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::model::network::TradingModel;
    pub use crate::continual::algorithm::ContinualMAMLTrainer;
    pub use crate::data::bybit::BybitClient;
    pub use crate::data::features::FeatureGenerator;
    pub use crate::trading::strategy::TradingStrategy;
    pub use crate::trading::signals::TradingSignal;
    pub use crate::backtest::engine::BacktestEngine;
}

/// Error types for the crate
#[derive(thiserror::Error, Debug)]
pub enum CMLError {
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

    #[error("Continual learning error: {0}")]
    ContinualError(String),
}

pub type Result<T> = std::result::Result<T, CMLError>;
