//! # Flow Models Trading
//!
//! This library provides implementations for Normalizing Flow models
//! applied to cryptocurrency trading using data from Bybit exchange.
//!
//! ## Core Concepts
//!
//! - **Normalizing Flows**: Invertible transformations for exact likelihood
//! - **Anomaly Detection**: Identify unusual market states via low likelihood
//! - **Regime Detection**: Cluster latent representations for market states
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `flow` - Normalizing flow layers and models
//! - `features` - Feature engineering from market data
//! - `strategy` - Trading signal generation
//! - `backtest` - Backtesting framework
//!
//! ## Example
//!
//! ```rust,no_run
//! use flow_models_trading::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch market data
//!     let client = BybitClient::new();
//!     let ohlcv = client.get_klines("BTCUSDT", "60", 500).await?;
//!
//!     // Build features
//!     let features = FeatureEngine::compute_features(&ohlcv);
//!
//!     // Create and train flow model
//!     let config = FlowConfig::default();
//!     let mut model = NormalizingFlow::new(config);
//!     model.train(&features, 100)?;
//!
//!     // Detect anomalies
//!     let anomaly_detector = AnomalyDetector::new(&model, 5.0);
//!     let is_anomaly = anomaly_detector.detect(&features.row(0))?;
//!
//!     // Generate trading signals
//!     let strategy = FlowTradingStrategy::new(model);
//!     let signal = strategy.generate_signal(&features.row(0))?;
//!
//!     println!("Signal: {:?}", signal);
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod backtest;
pub mod features;
pub mod flow;
pub mod strategy;

// Re-export commonly used types
pub use api::client::BybitClient;
pub use api::types::{Kline, OrderBook, Ticker, Trade};
pub use backtest::engine::BacktestEngine;
pub use backtest::report::BacktestReport;
pub use features::engine::FeatureEngine;
pub use features::indicators::TechnicalIndicators;
pub use flow::config::FlowConfig;
pub use flow::layers::{ActNorm, AffineCoupling, Permutation};
pub use flow::model::NormalizingFlow;
pub use flow::anomaly::AnomalyDetector;
pub use flow::regime::RegimeDetector;
pub use strategy::signal::{Signal, SignalGenerator, SignalType};
pub use strategy::flow_strategy::FlowTradingStrategy;

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
    pub use crate::api::types::{Kline, OrderBook, Ticker, Trade};
    pub use crate::backtest::engine::BacktestEngine;
    pub use crate::backtest::report::BacktestReport;
    pub use crate::features::engine::FeatureEngine;
    pub use crate::features::indicators::TechnicalIndicators;
    pub use crate::flow::config::FlowConfig;
    pub use crate::flow::layers::{ActNorm, AffineCoupling, Permutation};
    pub use crate::flow::model::NormalizingFlow;
    pub use crate::flow::anomaly::AnomalyDetector;
    pub use crate::flow::regime::RegimeDetector;
    pub use crate::strategy::signal::{Signal, SignalGenerator, SignalType};
    pub use crate::strategy::flow_strategy::FlowTradingStrategy;
    pub use crate::DEFAULT_SYMBOLS;
}
