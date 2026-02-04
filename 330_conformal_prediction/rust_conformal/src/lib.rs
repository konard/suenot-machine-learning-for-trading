//! # Conformal Prediction Trading
//!
//! This library provides implementations for Conformal Prediction methods
//! applied to cryptocurrency trading using data from Bybit exchange.
//!
//! ## Core Concepts
//!
//! - **Conformal Prediction**: Distribution-free prediction intervals with coverage guarantees
//! - **Split Conformal**: Efficient calibration using held-out data
//! - **Adaptive Conformal**: Online adjustment for time series
//! - **Uncertainty-based Trading**: Position sizing based on prediction confidence
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `conformal` - Conformal prediction algorithms
//! - `features` - Feature engineering from market data
//! - `strategy` - Trading signal generation with uncertainty
//! - `backtest` - Backtesting framework
//!
//! ## Example
//!
//! ```rust,no_run
//! use conformal_trading::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch market data
//!     let client = BybitClient::new();
//!     let klines = client.get_klines("BTCUSDT", "1h", 1000).await?;
//!
//!     // Create features
//!     let features = FeatureEngine::new(5).create_features(&klines);
//!
//!     // Create conformal predictor
//!     let mut predictor = SplitConformalPredictor::new(0.1);
//!     predictor.calibrate(&cal_scores);
//!
//!     // Generate trading signals
//!     let signal_gen = SignalGenerator::new(0.6, 1.0);
//!     let signals = signal_gen.generate(&predictions, &intervals);
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod backtest;
pub mod conformal;
pub mod features;
pub mod strategy;

// Re-export commonly used types
pub use api::client::BybitClient;
pub use api::types::{Kline, Ticker};
pub use backtest::engine::BacktestEngine;
pub use backtest::metrics::PerformanceMetrics;
pub use conformal::predictor::SplitConformalPredictor;
pub use conformal::adaptive::AdaptiveConformalPredictor;
pub use conformal::scores::NonConformityScore;
pub use features::engine::FeatureEngine;
pub use features::indicators::TechnicalIndicators;
pub use strategy::signal::{Signal, SignalGenerator, SignalType};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default trading symbols
pub const DEFAULT_SYMBOLS: &[&str] = &[
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT",
];

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::api::client::BybitClient;
    pub use crate::api::types::{Kline, Ticker};
    pub use crate::backtest::engine::BacktestEngine;
    pub use crate::backtest::metrics::PerformanceMetrics;
    pub use crate::conformal::predictor::SplitConformalPredictor;
    pub use crate::conformal::adaptive::AdaptiveConformalPredictor;
    pub use crate::conformal::scores::NonConformityScore;
    pub use crate::features::engine::FeatureEngine;
    pub use crate::features::indicators::TechnicalIndicators;
    pub use crate::strategy::signal::{Signal, SignalGenerator, SignalType};
    pub use crate::DEFAULT_SYMBOLS;
}
