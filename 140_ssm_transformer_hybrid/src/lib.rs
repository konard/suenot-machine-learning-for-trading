//! # SSM-Transformer Hybrid for Trading
//!
//! This crate implements hybrid State Space Model (SSM) and Transformer
//! architectures for algorithmic trading. The hybrid approach combines
//! the linear-time sequence processing of SSMs (Mamba-style) with the
//! global attention mechanism of Transformers.
//!
//! ## Features
//!
//! - SSM (Mamba-style) blocks for long-range dependency capture
//! - Transformer blocks for local pattern recognition
//! - Interleaved hybrid architecture (configurable ratio)
//! - Multi-task prediction: direction, volatility, return magnitude
//! - Bybit API integration for cryptocurrency data
//! - Feature engineering pipeline (RSI, MACD, Bollinger Bands, ATR, OBV)
//! - Backtesting framework for strategy evaluation
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use ssm_transformer_hybrid::{SSMTransformerModel, BybitClient, BacktestEngine};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let model = SSMTransformerModel::new(15, 64, 6, 2, 16, 4);
//!     let client = BybitClient::new();
//!     let data = client.fetch_klines("BTCUSDT", "60", 1000).await?;
//!
//!     let engine = BacktestEngine::new(100_000.0, 0.001);
//!     let results = engine.run(&model, &data)?;
//!     println!("Sharpe: {:.3}", results.sharpe_ratio);
//!     Ok(())
//! }
//! ```

pub mod model;
pub mod data;
pub mod trading;
pub mod backtest;

pub use model::ssm_block::SSMBlock;
pub use model::transformer_block::TransformerBlock;
pub use model::hybrid::SSMTransformerModel;
pub use data::bybit::BybitClient;
pub use data::features::FeatureGenerator;
pub use trading::strategy::TradingStrategy;
pub use trading::signals::TradingSignal;
pub use backtest::engine::BacktestEngine;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::model::hybrid::SSMTransformerModel;
    pub use crate::data::bybit::BybitClient;
    pub use crate::data::features::FeatureGenerator;
    pub use crate::trading::strategy::TradingStrategy;
    pub use crate::trading::signals::TradingSignal;
    pub use crate::backtest::engine::BacktestEngine;
}

/// Error types for the crate
#[derive(thiserror::Error, Debug)]
pub enum HybridError {
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
}

pub type Result<T> = std::result::Result<T, HybridError>;
