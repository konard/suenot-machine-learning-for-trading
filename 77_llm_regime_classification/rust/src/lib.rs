//! Chapter 77: LLM Regime Classification
//!
//! This crate provides tools for classifying market regimes using
//! statistical methods and Large Language Model (LLM) approaches.
//!
//! # Overview
//!
//! Market regime classification helps identify the current state of the market:
//! - Bull: Upward trending with positive momentum
//! - Bear: Downward trending with negative momentum
//! - Sideways: Range-bound with no clear direction
//! - HighVolatility: Elevated volatility regardless of direction
//! - Crisis: Extreme market stress conditions
//!
//! # Example
//!
//! ```rust,ignore
//! use regime_classification::{
//!     data::YahooFinanceLoader,
//!     classifier::{MarketRegime, StatisticalClassifier},
//!     signals::SignalGenerator,
//! };
//!
//! #[tokio::main]
//! async fn main() {
//!     // Load market data
//!     let loader = YahooFinanceLoader::new();
//!     let data = loader.get_daily("SPY", "1y").await.unwrap();
//!
//!     // Classify regime
//!     let classifier = StatisticalClassifier::new(20, 0.02, 0.001);
//!     let result = classifier.classify(&data);
//!
//!     println!("Current regime: {:?}", result.regime);
//! }
//! ```

pub mod data;
pub mod classifier;
pub mod signals;
pub mod backtest;
pub mod evaluate;

pub use classifier::{MarketRegime, RegimeResult};
pub use data::{OHLCVData, DataLoader};
pub use signals::{TradingSignal, SignalType, SignalGenerator};
pub use backtest::{BacktestResult, Backtester};
