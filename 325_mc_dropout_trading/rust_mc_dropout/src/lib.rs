//! # MC Dropout Trading
//!
//! This library provides implementations for Monte Carlo Dropout
//! applied to cryptocurrency trading using data from Bybit exchange.
//!
//! MC Dropout enables uncertainty estimation in neural network predictions,
//! which is crucial for making informed trading decisions.
//!
//! ## Core Concepts
//!
//! - **MC Dropout**: Keep dropout active during inference for uncertainty estimation
//! - **Bayesian Approximation**: Dropout approximates Bayesian inference
//! - **Uncertainty-Aware Trading**: Use prediction confidence for position sizing
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `model` - MC Dropout neural network implementation
//! - `features` - Feature engineering from market data
//! - `strategy` - Trading signal generation with uncertainty
//! - `backtest` - Backtesting framework
//!
//! ## Example
//!
//! ```rust,no_run
//! use mc_dropout_trading::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch market data
//!     let client = BybitClient::new();
//!     let ticker = client.get_ticker("BTCUSDT").await?;
//!
//!     // Create features
//!     let features = FeatureEngine::new().compute(&ticker);
//!
//!     // Run MC Dropout inference
//!     let model = MCDropoutModel::new(MCDropoutConfig::default());
//!     let prediction = model.predict_with_uncertainty(&features, 50);
//!
//!     // Generate signal
//!     let signal_generator = SignalGenerator::new(SignalConfig::default());
//!     let signal = signal_generator.generate(&prediction);
//!
//!     println!("Signal: {:?}", signal);
//!     println!("Confidence: {:.2}", signal.confidence);
//!     println!("Position Size: {:.4}", signal.position_size);
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod backtest;
pub mod features;
pub mod model;
pub mod strategy;

// Re-export commonly used types
pub use api::client::BybitClient;
pub use api::types::{Kline, OrderBook, Ticker, Trade};
pub use backtest::engine::BacktestEngine;
pub use backtest::report::BacktestReport;
pub use features::engine::FeatureEngine;
pub use features::indicators::TechnicalIndicators;
pub use model::config::MCDropoutConfig;
pub use model::dropout::MCDropoutModel;
pub use model::prediction::PredictionResult;
pub use strategy::config::SignalConfig;
pub use strategy::signal::{Signal, SignalGenerator, SignalType};

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
    pub use crate::model::config::MCDropoutConfig;
    pub use crate::model::dropout::MCDropoutModel;
    pub use crate::model::prediction::PredictionResult;
    pub use crate::strategy::config::SignalConfig;
    pub use crate::strategy::signal::{Signal, SignalGenerator, SignalType};
    pub use crate::DEFAULT_SYMBOLS;
}
