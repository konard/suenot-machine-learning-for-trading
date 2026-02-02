//! # Transfer Entropy for Trading
//!
//! This crate implements Transfer Entropy (TE) analysis for algorithmic trading.
//! TE measures directed information flow between time series, enabling detection
//! of lead-lag relationships between assets for predictive signal construction.
//!
//! ## Features
//!
//! - Multiple TE estimation methods (binning, symbolic)
//! - Effective TE with permutation testing
//! - Directed information flow network construction
//! - Leader-follower identification
//! - Bybit API integration for cryptocurrency data
//! - Backtesting framework for TE-based strategies
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use te_trading::{TransferEntropyEstimator, TEMethod, BybitClient};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let estimator = TransferEntropyEstimator::new(
//!         TEMethod::Binning { n_bins: 8 },
//!         3,
//!     );
//!
//!     let client = BybitClient::new();
//!     let data = client.fetch_klines("BTCUSDT", "60", 1000).await?;
//!
//!     Ok(())
//! }
//! ```

pub mod entropy;
pub mod network;
pub mod data;
pub mod trading;
pub mod backtest;

pub use entropy::estimator::TransferEntropyEstimator;
pub use entropy::estimator::TEMethod;
pub use network::builder::TENetwork;
pub use data::bybit::BybitClient;
pub use data::simulation::SimulatedDataGenerator;
pub use trading::strategy::TEStrategy;
pub use trading::signals::TradingSignal;
pub use backtest::engine::BacktestEngine;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::entropy::estimator::{TransferEntropyEstimator, TEMethod};
    pub use crate::network::builder::TENetwork;
    pub use crate::data::bybit::BybitClient;
    pub use crate::data::simulation::SimulatedDataGenerator;
    pub use crate::trading::strategy::TEStrategy;
    pub use crate::trading::signals::TradingSignal;
    pub use crate::backtest::engine::BacktestEngine;
}

/// Error types for the crate
#[derive(thiserror::Error, Debug)]
pub enum TEError {
    #[error("Estimation error: {0}")]
    EstimationError(String),

    #[error("Data error: {0}")]
    DataError(String),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Backtest error: {0}")]
    BacktestError(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Network error: {0}")]
    NetworkError(String),
}

pub type Result<T> = std::result::Result<T, TEError>;
