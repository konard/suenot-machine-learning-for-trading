//! # Meta-Volatility Prediction for Trading
//!
//! This crate implements MAML-based meta-learning for volatility prediction
//! in algorithmic trading. The model learns to quickly adapt its volatility
//! forecasts to new assets and market regimes with minimal data.
//!
//! ## Features
//!
//! - MAML (Model-Agnostic Meta-Learning) for fast volatility adaptation
//! - Neural volatility estimator with Softplus output
//! - Bybit API integration for cryptocurrency data
//! - Feature engineering (realized volatility, RSI, Bollinger Bands)
//! - Backtesting framework with volatility-adaptive position sizing
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use meta_volatility::{MetaVolatilityNet, MAMLTrainer, BybitClient};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let net = MetaVolatilityNet::new(10, 64);
//!     let mut trainer = MAMLTrainer::new(net, 0.01, 0.001, 5);
//!
//!     let client = BybitClient::new();
//!     let data = client.fetch_klines("BTCUSDT", "60", 1000).await?;
//!
//!     Ok(())
//! }
//! ```

pub mod model;
pub mod meta;
pub mod data;
pub mod trading;
pub mod backtest;

pub use model::network::MetaVolatilityNet;
pub use meta::maml::MAMLTrainer;
pub use data::bybit::BybitClient;
pub use data::features::FeatureGenerator;
pub use trading::strategy::VolatilityStrategy;
pub use trading::signals::TradingSignal;
pub use backtest::engine::BacktestEngine;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::model::network::MetaVolatilityNet;
    pub use crate::meta::maml::MAMLTrainer;
    pub use crate::data::bybit::BybitClient;
    pub use crate::data::features::FeatureGenerator;
    pub use crate::trading::strategy::VolatilityStrategy;
    pub use crate::trading::signals::TradingSignal;
    pub use crate::backtest::engine::BacktestEngine;
}

/// Error types for the crate
#[derive(thiserror::Error, Debug)]
pub enum MetaVolatilityError {
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

    #[error("Meta-learning error: {0}")]
    MetaLearningError(String),
}

pub type Result<T> = std::result::Result<T, MetaVolatilityError>;
