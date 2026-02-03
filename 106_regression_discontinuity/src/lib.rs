//! # Regression Discontinuity Design for Trading
//!
//! This crate implements Regression Discontinuity Design (RDD) methods for
//! algorithmic trading. RDD is a quasi-experimental method that exploits
//! naturally occurring thresholds to identify causal effects and trading
//! opportunities.
//!
//! ## Features
//!
//! - Sharp and Fuzzy RDD estimation with local linear regression
//! - Optimal bandwidth selection (IK and CCT methods)
//! - Kernel weighting (triangular, uniform, Epanechnikov)
//! - Bybit API integration for cryptocurrency data
//! - RSI and other technical indicator thresholds
//! - Backtesting framework for RDD strategies
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use rdd_trading::{RDDModel, BybitClient, BacktestEngine};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let client = BybitClient::new();
//!     let data = client.fetch_klines("BTCUSDT", "60", 1000).await?;
//!
//!     let model = RDDModel::new(30.0, 5.0, Kernel::Triangular);
//!     let rsi = compute_rsi(&data.close_prices(), 14);
//!     let returns = data.returns();
//!
//!     let results = model.fit(&rsi, &returns)?;
//!     println!("Treatment Effect: {:.4}", results.tau);
//!     Ok(())
//! }
//! ```
//!
//! ## References
//!
//! - Imbens & Lemieux (2008), "Regression Discontinuity Designs: A Guide to Practice"
//! - Calonico, Cattaneo & Titiunik (2014), "Robust Nonparametric Confidence Intervals"

pub mod model;
pub mod data;
pub mod trading;
pub mod backtest;

pub use model::rdd::{RDDModel, RDDResults, Kernel};
pub use model::validator::{RDDValidator, DensityTestResult, PlaceboTestResult};
pub use data::bybit::{BybitClient, MarketData, Candle};
pub use data::indicators::{compute_rsi, compute_sma, compute_ema};
pub use trading::strategy::RDDStrategy;
pub use trading::signals::{TradingSignal, SignalStrength};
pub use backtest::engine::BacktestEngine;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::model::rdd::{RDDModel, RDDResults, Kernel};
    pub use crate::model::validator::{RDDValidator, DensityTestResult};
    pub use crate::data::bybit::{BybitClient, MarketData};
    pub use crate::data::indicators::{compute_rsi, compute_sma};
    pub use crate::trading::strategy::RDDStrategy;
    pub use crate::backtest::engine::BacktestEngine;
}

/// Error types for the crate.
#[derive(thiserror::Error, Debug)]
pub enum RDDError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("JSON parsing failed: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Insufficient data: need {needed}, got {got}")]
    InsufficientData { needed: usize, got: usize },

    #[error("Computation error: {0}")]
    ComputationError(String),

    #[error("Model not fitted")]
    NotFitted,
}

pub type Result<T> = std::result::Result<T, RDDError>;
