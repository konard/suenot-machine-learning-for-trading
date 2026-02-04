//! # Deep Ensembles Trading
//!
//! This library provides implementations for Deep Ensemble models
//! applied to cryptocurrency trading using data from Bybit exchange.
//!
//! ## Core Concepts
//!
//! - **Deep Ensembles**: Multiple neural networks for uncertainty estimation
//! - **Epistemic Uncertainty**: Model disagreement (reducible with more data)
//! - **Aleatoric Uncertainty**: Data noise (irreducible)
//! - **NLL Loss**: Proper scoring rule for uncertainty estimation
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `ensemble` - Deep Ensemble model implementation
//! - `features` - Feature engineering from market data
//! - `strategy` - Trading signal generation with uncertainty
//! - `backtest` - Backtesting framework
//!
//! ## Example
//!
//! ```rust,no_run
//! use deep_ensembles_trading::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch market data
//!     let client = BybitClient::new();
//!     let klines = client.get_klines("BTCUSDT", "1h", 500).await?;
//!
//!     // Prepare features
//!     let features = FeatureEngine::compute_features(&klines);
//!
//!     // Create and train ensemble
//!     let config = EnsembleConfig::default();
//!     let mut ensemble = DeepEnsemble::new(config);
//!     ensemble.train(&features, &targets, 100)?;
//!
//!     // Generate signals with uncertainty
//!     let strategy = UncertaintyStrategy::new(0.6);
//!     let signals = strategy.generate(&ensemble, &features);
//!
//!     for signal in signals {
//!         println!("{:?}", signal);
//!     }
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod backtest;
pub mod ensemble;
pub mod features;
pub mod strategy;

// Re-export commonly used types
pub use api::client::BybitClient;
pub use api::types::{Kline, OrderBook, Ticker};
pub use backtest::engine::BacktestEngine;
pub use backtest::report::BacktestReport;
pub use ensemble::config::EnsembleConfig;
pub use ensemble::model::DeepEnsemble;
pub use ensemble::prediction::EnsemblePrediction;
pub use features::engine::FeatureEngine;
pub use features::indicators::TechnicalIndicators;
pub use strategy::signal::{Signal, SignalGenerator, SignalType};
pub use strategy::uncertainty::UncertaintyStrategy;

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
    pub use crate::ensemble::config::EnsembleConfig;
    pub use crate::ensemble::model::DeepEnsemble;
    pub use crate::ensemble::prediction::EnsemblePrediction;
    pub use crate::features::engine::FeatureEngine;
    pub use crate::features::indicators::TechnicalIndicators;
    pub use crate::strategy::signal::{Signal, SignalGenerator, SignalType};
    pub use crate::strategy::uncertainty::UncertaintyStrategy;
    pub use crate::DEFAULT_SYMBOLS;
}
