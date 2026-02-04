//! # Variational Inference Trading
//!
//! This library provides implementations for Variational Inference models
//! applied to cryptocurrency trading using data from Bybit exchange.
//!
//! ## Core Concepts
//!
//! - **Variational Autoencoder (VAE)**: Learn latent representations of market states
//! - **Uncertainty Estimation**: Quantify prediction confidence
//! - **Trading Signals**: Generate signals with uncertainty-aware position sizing
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `variational` - VAE and variational inference implementations
//! - `features` - Feature engineering from market data
//! - `strategy` - Trading signal generation
//! - `backtest` - Backtesting framework
//!
//! ## Example
//!
//! ```rust,no_run
//! use variational_inference_trading::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch market data
//!     let client = BybitClient::new();
//!     let klines = client.get_klines("BTCUSDT", "1", 100).await?;
//!
//!     // Build features
//!     let feature_engine = FeatureEngine::new();
//!     let features = feature_engine.compute(&klines)?;
//!
//!     // Run VAE inference
//!     let config = VAEConfig::default();
//!     let vae = VAE::new(config);
//!     let prediction = vae.predict(&features);
//!
//!     // Generate signals
//!     let signal_generator = SignalGenerator::new(0.6);
//!     let signal = signal_generator.generate(&prediction);
//!
//!     println!("{:?}", signal);
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod backtest;
pub mod features;
pub mod strategy;
pub mod variational;

// Re-export commonly used types
pub use api::client::BybitClient;
pub use api::types::{Kline, OrderBook, Ticker, Trade};
pub use backtest::engine::BacktestEngine;
pub use backtest::report::BacktestReport;
pub use features::engine::FeatureEngine;
pub use features::indicators::TechnicalIndicators;
pub use strategy::signal::{Signal, SignalGenerator, SignalType};
pub use variational::config::VAEConfig;
pub use variational::encoder::Encoder;
pub use variational::decoder::Decoder;
pub use variational::vae::VAE;
pub use variational::prediction::Prediction;

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
    pub use crate::strategy::signal::{Signal, SignalGenerator, SignalType};
    pub use crate::variational::config::VAEConfig;
    pub use crate::variational::encoder::Encoder;
    pub use crate::variational::decoder::Decoder;
    pub use crate::variational::vae::VAE;
    pub use crate::variational::prediction::Prediction;
    pub use crate::DEFAULT_SYMBOLS;
}
