//! # Bayesian Neural Network Trading
//!
//! This library provides implementations for Bayesian Neural Networks
//! applied to cryptocurrency trading using data from Bybit exchange.
//!
//! ## Core Concepts
//!
//! - **Weight Uncertainty**: Weights are distributions, not point estimates
//! - **Epistemic Uncertainty**: Model uncertainty that decreases with more data
//! - **Aleatoric Uncertainty**: Irreducible data noise
//! - **Uncertainty-Aware Trading**: Position sizing based on confidence
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `bnn` - Bayesian Neural Network implementation
//! - `features` - Feature engineering from market data
//! - `strategy` - Uncertainty-aware trading strategy
//! - `backtest` - Backtesting framework
//!
//! ## Example
//!
//! ```rust,no_run
//! use bayesian_nn_trading::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch market data
//!     let client = BybitClient::new();
//!     let klines = client.get_klines("BTCUSDT", "1h", 1000).await?;
//!
//!     // Create features
//!     let features = FeatureEngine::new().create_features(&klines)?;
//!
//!     // Initialize BNN
//!     let config = BNNConfig::default();
//!     let mut bnn = BayesianNN::new(config);
//!
//!     // Train and predict
//!     bnn.train(&features.x, &features.y)?;
//!     let prediction = bnn.predict_with_uncertainty(&features.x)?;
//!
//!     println!("Prediction: {:?}", prediction);
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod backtest;
pub mod bnn;
pub mod features;
pub mod strategy;

// Re-export commonly used types
pub use api::client::BybitClient;
pub use api::types::{Kline, OrderBook, Ticker};
pub use backtest::engine::BacktestEngine;
pub use backtest::report::BacktestReport;
pub use bnn::config::BNNConfig;
pub use bnn::layers::BayesianLinear;
pub use bnn::model::BayesianNN;
pub use bnn::uncertainty::{UncertaintyEstimate, UncertaintyType};
pub use features::engine::FeatureEngine;
pub use features::indicators::TechnicalIndicators;
pub use strategy::signal::{Signal, SignalGenerator, SignalType};
pub use strategy::trading::TradingStrategy;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default trading symbols for examples
pub const DEFAULT_SYMBOLS: &[&str] = &[
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT",
];

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::api::client::BybitClient;
    pub use crate::api::types::{Kline, OrderBook, Ticker};
    pub use crate::backtest::engine::BacktestEngine;
    pub use crate::backtest::report::BacktestReport;
    pub use crate::bnn::config::BNNConfig;
    pub use crate::bnn::layers::BayesianLinear;
    pub use crate::bnn::model::BayesianNN;
    pub use crate::bnn::uncertainty::{UncertaintyEstimate, UncertaintyType};
    pub use crate::features::engine::FeatureEngine;
    pub use crate::features::indicators::TechnicalIndicators;
    pub use crate::strategy::signal::{Signal, SignalGenerator, SignalType};
    pub use crate::strategy::trading::TradingStrategy;
    pub use crate::DEFAULT_SYMBOLS;
}
