//! # Zero-Shot Trading for Cryptocurrency Markets
//!
//! This crate implements zero-shot learning for algorithmic trading,
//! enabling predictions on new assets without task-specific training.
//!
//! ## Features
//!
//! - Zero-shot regime prediction via attribute matching
//! - Bybit API integration for cryptocurrency data
//! - Feature engineering for market time series
//! - Backtesting framework for strategy evaluation
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use zero_shot_trading::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create model and strategy
//!     let model = ZeroShotModel::new(11, 64);
//!     let strategy = TradingStrategy::new(model, 0.6);
//!
//!     // Fetch data from Bybit
//!     let client = BybitClient::new();
//!     let klines = client.fetch_klines("BTCUSDT", "1h", 200).await?;
//!
//!     // Generate trading signal
//!     let features = prepare_features(&klines)?;
//!     let signal = strategy.generate_signal(&features)?;
//!
//!     println!("Regime: {:?}", signal.regime);
//!     println!("Confidence: {:.1}%", signal.confidence * 100.0);
//!
//!     Ok(())
//! }
//! ```

pub mod model;
pub mod data;
pub mod training;
pub mod strategy;
pub mod backtest;

pub use model::network::ZeroShotModel;
pub use model::embeddings::EmbeddingSpace;
pub use data::bybit::BybitClient;
pub use data::features::{FeatureGenerator, prepare_features};
pub use data::attributes::{AssetAttributes, AssetRegistry};
pub use strategy::regime::{RegimeDetector, ZeroShotRegimePredictor};
pub use strategy::signals::{Signal, TimestampedSignal, SignalFilter};
pub use strategy::trading::{ZeroShotTradingStrategy, StrategyConfig, Position};
pub use backtest::engine::{BacktestEngine, BacktestConfig};
pub use backtest::results::{BacktestResults, Trade};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::model::network::ZeroShotModel;
    pub use crate::model::embeddings::EmbeddingSpace;
    pub use crate::data::bybit::BybitClient;
    pub use crate::data::features::{FeatureGenerator, prepare_features};
    pub use crate::data::attributes::{AssetAttributes, AssetRegistry};
    pub use crate::strategy::regime::{RegimeDetector, ZeroShotRegimePredictor};
    pub use crate::strategy::signals::{Signal, TimestampedSignal, SignalFilter};
    pub use crate::strategy::trading::{ZeroShotTradingStrategy, StrategyConfig, Position};
    pub use crate::backtest::engine::{BacktestEngine, BacktestConfig};
    pub use crate::backtest::results::{BacktestResults, Trade};
    pub use crate::{MarketRegime, Result, ZeroShotError};
}

/// Error types for the crate
#[derive(thiserror::Error, Debug)]
pub enum ZeroShotError {
    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Data error: {0}")]
    DataError(String),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Feature error: {0}")]
    FeatureError(String),

    #[error("Strategy error: {0}")]
    StrategyError(String),

    #[error("Backtest error: {0}")]
    BacktestError(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Not enough data: need {needed}, got {got}")]
    NotEnoughData { needed: usize, got: usize },
}

pub type Result<T> = std::result::Result<T, ZeroShotError>;

/// Market regime types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MarketRegime {
    StrongUptrend,
    WeakUptrend,
    Sideways,
    WeakDowntrend,
    StrongDowntrend,
}

impl MarketRegime {
    /// Get regime name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            MarketRegime::StrongUptrend => "strong_uptrend",
            MarketRegime::WeakUptrend => "weak_uptrend",
            MarketRegime::Sideways => "sideways",
            MarketRegime::WeakDowntrend => "weak_downtrend",
            MarketRegime::StrongDowntrend => "strong_downtrend",
        }
    }

    /// Get all regimes
    pub fn all() -> &'static [MarketRegime] {
        &[
            MarketRegime::StrongUptrend,
            MarketRegime::WeakUptrend,
            MarketRegime::Sideways,
            MarketRegime::WeakDowntrend,
            MarketRegime::StrongDowntrend,
        ]
    }

    /// Get base position size for regime
    pub fn base_position(&self) -> f64 {
        match self {
            MarketRegime::StrongUptrend => 1.0,
            MarketRegime::WeakUptrend => 0.5,
            MarketRegime::Sideways => 0.0,
            MarketRegime::WeakDowntrend => -0.5,
            MarketRegime::StrongDowntrend => -1.0,
        }
    }
}

impl std::fmt::Display for MarketRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
