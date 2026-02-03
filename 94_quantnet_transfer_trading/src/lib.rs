//! # QuantNet Transfer Trading
//!
//! This crate implements QuantNet-style transfer learning for algorithmic trading.
//! QuantNet uses an encoder-decoder architecture with asset-specific trading heads,
//! enabling knowledge transfer across multiple assets through a shared latent space.
//!
//! ## Features
//!
//! - Encoder-decoder architecture with GELU activation
//! - Asset-specific trading heads for multi-asset signals
//! - Xavier weight initialization
//! - Sharpe ratio loss for direct portfolio optimization
//! - Pretrain/finetune two-phase training
//! - Bybit API integration for cryptocurrency data
//! - Feature engineering pipeline
//! - Backtesting framework for strategy evaluation
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use quantnet_trading::{QuantNetModel, QuantNetTrainer, BybitClient};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let model = QuantNetModel::new(10, 32, 16, &["BTCUSDT", "ETHUSDT"]);
//!     let trainer = QuantNetTrainer::new(model, 0.001, 1.0);
//!
//!     let client = BybitClient::new();
//!     let data = client.fetch_klines("BTCUSDT", "60", 1000).await?;
//!
//!     Ok(())
//! }
//! ```

pub mod model;
pub mod data;
pub mod trading;
pub mod backtest;
pub mod quantnet;

pub use model::network::QuantNetModel;
pub use quantnet::trainer::QuantNetTrainer;
pub use data::bybit::BybitClient;
pub use data::features::FeatureGenerator;
pub use trading::strategy::TradingStrategy;
pub use trading::signals::TradingSignal;
pub use backtest::engine::BacktestEngine;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::model::network::QuantNetModel;
    pub use crate::quantnet::trainer::QuantNetTrainer;
    pub use crate::data::bybit::BybitClient;
    pub use crate::data::features::FeatureGenerator;
    pub use crate::trading::strategy::TradingStrategy;
    pub use crate::trading::signals::TradingSignal;
    pub use crate::backtest::engine::BacktestEngine;
}

/// Error types for the crate
#[derive(thiserror::Error, Debug)]
pub enum QuantNetError {
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

pub type Result<T> = std::result::Result<T, QuantNetError>;
