//! # Continual Meta-Learning for Trading
//!
//! This crate implements the Continual Meta-Learning (CML) algorithm for algorithmic trading.
//! CML combines meta-learning's rapid adaptation with continual learning's ability to
//! retain knowledge across changing market regimes.
//!
//! ## Features
//!
//! - Experience Replay for memory-based learning
//! - Elastic Weight Consolidation (EWC) to prevent forgetting
//! - Fast adaptation to new market conditions
//! - Bybit API integration for cryptocurrency data
//! - Backtesting framework for strategy evaluation
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use continual_meta_learning::{ContinualMetaLearner, TradingModel, BybitClient, CMLConfig};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create model and trainer
//!     let config = CMLConfig::default();
//!     let learner = ContinualMetaLearner::new(config);
//!
//!     // Fetch data
//!     let client = BybitClient::new();
//!     let data = client.get_klines("BTCUSDT", Interval::Hour1, Category::Linear, Some(1000), None, None).await?;
//!
//!     Ok(())
//! }
//! ```

pub mod model;
pub mod continual;
pub mod data;
pub mod trading;
pub mod backtest;

// Re-exports
pub use model::TradingModel;
pub use continual::learner::ContinualMetaLearner;
pub use continual::memory::{MemoryBuffer, Experience};
pub use continual::ewc::EWC;
pub use data::bybit::{BybitClient, Kline, Interval, Category};
pub use data::features::{TradingFeatures, FeatureConfig};
pub use trading::strategy::{CMLStrategy, StrategyConfig, Position, TradeAction};
pub use trading::signals::{Signal, SignalGenerator};
pub use backtest::engine::{Backtester, BacktestConfig, BacktestResult};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::model::TradingModel;
    pub use crate::continual::learner::ContinualMetaLearner;
    pub use crate::continual::memory::{MemoryBuffer, Experience};
    pub use crate::continual::ewc::EWC;
    pub use crate::data::bybit::{BybitClient, Kline, Interval, Category};
    pub use crate::data::features::{TradingFeatures, FeatureConfig};
    pub use crate::trading::strategy::{CMLStrategy, StrategyConfig};
    pub use crate::trading::signals::{Signal, SignalGenerator};
    pub use crate::backtest::engine::{Backtester, BacktestConfig, BacktestResult};
    pub use crate::{CMLConfig, CMLError, MarketRegime};
}

/// Configuration for CML training
#[derive(Debug, Clone)]
pub struct CMLConfig {
    /// Input feature dimension.
    pub input_size: usize,
    /// Hidden layer size.
    pub hidden_size: usize,
    /// Output dimension.
    pub output_size: usize,
    /// Learning rate for task-specific adaptation (inner loop).
    pub inner_lr: f64,
    /// Meta-learning rate (outer loop).
    pub outer_lr: f64,
    /// Number of SGD steps per task in inner loop.
    pub inner_steps: usize,
    /// Maximum number of experiences to store in memory.
    pub memory_size: usize,
    /// Strength of elastic weight consolidation.
    pub ewc_lambda: f64,
}

impl Default for CMLConfig {
    fn default() -> Self {
        Self {
            input_size: 9,
            hidden_size: 32,
            output_size: 1,
            inner_lr: 0.01,
            outer_lr: 0.001,
            inner_steps: 5,
            memory_size: 1000,
            ewc_lambda: 1000.0,
        }
    }
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

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("Backtest error: {0}")]
    BacktestError(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

pub type Result<T> = std::result::Result<T, CMLError>;

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MarketRegime {
    /// Bullish market (upward trend).
    Bull,
    /// Bearish market (downward trend).
    Bear,
    /// High volatility market.
    HighVolatility,
    /// Low volatility market.
    LowVolatility,
    /// Sideways/range-bound market.
    #[default]
    Sideways,
}

impl std::fmt::Display for MarketRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarketRegime::Bull => write!(f, "bull"),
            MarketRegime::Bear => write!(f, "bear"),
            MarketRegime::HighVolatility => write!(f, "high_vol"),
            MarketRegime::LowVolatility => write!(f, "low_vol"),
            MarketRegime::Sideways => write!(f, "sideways"),
        }
    }
}

impl From<&str> for MarketRegime {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "bull" | "bullish" => MarketRegime::Bull,
            "bear" | "bearish" => MarketRegime::Bear,
            "high_vol" | "high_volatility" => MarketRegime::HighVolatility,
            "low_vol" | "low_volatility" => MarketRegime::LowVolatility,
            "sideways" | "range" | "ranging" => MarketRegime::Sideways,
            _ => MarketRegime::Sideways,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_regime() {
        assert_eq!(MarketRegime::from("bull"), MarketRegime::Bull);
        assert_eq!(MarketRegime::from("BEAR"), MarketRegime::Bear);
        assert_eq!(MarketRegime::from("high_vol"), MarketRegime::HighVolatility);
        assert_eq!(MarketRegime::from("unknown"), MarketRegime::Sideways);

        assert_eq!(format!("{}", MarketRegime::Bull), "bull");
    }

    #[test]
    fn test_cml_config_default() {
        let config = CMLConfig::default();
        assert_eq!(config.input_size, 9);
        assert_eq!(config.inner_lr, 0.01);
        assert_eq!(config.ewc_lambda, 1000.0);
    }
}
