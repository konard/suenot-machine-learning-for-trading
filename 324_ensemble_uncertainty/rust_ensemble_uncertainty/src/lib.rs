//! # Ensemble Uncertainty Trading
//!
//! This library provides implementations for ensemble-based uncertainty
//! quantification in cryptocurrency trading using data from Bybit exchange.
//!
//! ## Core Concepts
//!
//! - **Ensemble Models**: Multiple models working together for robust predictions
//! - **Uncertainty Quantification**: Measuring prediction confidence
//! - **Trading Signals**: Uncertainty-aware trading decisions
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `ensemble` - Ensemble model implementations (Random Forest, Gradient Boosting)
//! - `uncertainty` - Uncertainty quantification methods
//! - `strategy` - Trading signal generation with uncertainty
//! - `backtest` - Backtesting framework
//!
//! ## Example
//!
//! ```rust,no_run
//! use ensemble_uncertainty_trading::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch market data
//!     let client = BybitClient::new();
//!     let klines = client.get_klines("BTCUSDT", "1h", 1000).await?;
//!
//!     // Compute features
//!     let features = FeatureEngine::compute_all(&klines);
//!
//!     // Train ensemble
//!     let ensemble = RandomForestEnsemble::new(100, 10);
//!     ensemble.fit(&features.x, &features.y)?;
//!
//!     // Get predictions with uncertainty
//!     let (predictions, uncertainties) = ensemble.predict_with_uncertainty(&features.x)?;
//!
//!     // Generate trading signals
//!     let strategy = UncertaintyStrategy::new(0.001, 0.03, 0.6);
//!     let signals = strategy.generate_signals(&predictions, &uncertainties);
//!
//!     for signal in signals.iter().take(5) {
//!         println!("{:?}", signal);
//!     }
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod backtest;
pub mod ensemble;
pub mod strategy;
pub mod uncertainty;

// Re-export commonly used types
pub use api::client::BybitClient;
pub use api::types::{Kline, OrderBook, Ticker, Trade};
pub use backtest::engine::BacktestEngine;
pub use backtest::report::BacktestReport;
pub use ensemble::random_forest::RandomForestEnsemble;
pub use ensemble::gradient_boosting::GradientBoostingEnsemble;
pub use ensemble::stacking::StackingEnsemble;
pub use strategy::signal::{Signal, SignalGenerator, SignalType};
pub use strategy::uncertainty_strategy::UncertaintyStrategy;
pub use uncertainty::quantifier::UncertaintyQuantifier;
pub use uncertainty::calibration::Calibrator;

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
    pub use crate::ensemble::random_forest::RandomForestEnsemble;
    pub use crate::ensemble::gradient_boosting::GradientBoostingEnsemble;
    pub use crate::ensemble::stacking::StackingEnsemble;
    pub use crate::ensemble::traits::EnsembleModel;
    pub use crate::strategy::signal::{Signal, SignalGenerator, SignalType};
    pub use crate::strategy::uncertainty_strategy::UncertaintyStrategy;
    pub use crate::uncertainty::quantifier::UncertaintyQuantifier;
    pub use crate::uncertainty::calibration::Calibrator;
    pub use crate::DEFAULT_SYMBOLS;
}
