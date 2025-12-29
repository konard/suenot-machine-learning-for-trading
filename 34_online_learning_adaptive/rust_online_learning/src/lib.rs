//! Online Learning Library for Cryptocurrency Trading
//!
//! This library provides implementations of online learning algorithms
//! for adaptive trading strategies with cryptocurrency data from Bybit.
//!
//! # Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `models` - Online learning models (linear regression, adaptive weights)
//! - `drift` - Concept drift detection algorithms (ADWIN, DDM)
//! - `features` - Feature engineering for momentum factors
//! - `streaming` - Streaming data simulation
//! - `backtest` - Backtesting framework for online strategies
//!
//! # Example
//!
//! ```rust,no_run
//! use online_learning::api::BybitClient;
//! use online_learning::models::OnlineLinearRegression;
//! use online_learning::drift::ADWIN;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Fetch data
//!     let client = BybitClient::new();
//!     let candles = client.get_klines("BTCUSDT", "1h", 500).await.unwrap();
//!
//!     // Create online model with drift detection
//!     let mut model = OnlineLinearRegression::new(5, 0.01);
//!     let mut drift_detector = ADWIN::new(0.002);
//!
//!     // Stream and learn
//!     for candle in &candles {
//!         // ... process and learn
//!     }
//! }
//! ```

pub mod api;
pub mod backtest;
pub mod drift;
pub mod features;
pub mod models;
pub mod streaming;

// Re-export commonly used types
pub use api::BybitClient;
pub use backtest::BacktestEngine;
pub use drift::{DriftDetector, ADWIN, DDM};
pub use features::MomentumFeatures;
pub use models::{AdaptiveMomentumWeights, OnlineLinearRegression};
pub use streaming::StreamSimulator;
