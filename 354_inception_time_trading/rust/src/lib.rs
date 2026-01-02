//! # InceptionTime for Cryptocurrency Trading
//!
//! This crate provides a modular implementation of InceptionTime architecture
//! for time series classification applied to cryptocurrency trading using Bybit data.
//!
//! ## Modules
//!
//! - `data`: Data fetching from Bybit API, preprocessing, and feature engineering
//! - `model`: InceptionTime architecture (Inception modules, network, ensemble)
//! - `training`: Training loop, loss functions, and evaluation metrics
//! - `strategy`: Trading signal generation, position management, and risk control
//! - `backtest`: Backtesting engine and performance analytics
//! - `utils`: Configuration, logging, and helper functions
//!
//! ## Example
//!
//! ```no_run
//! use inception_time_trading::{BybitClient, InceptionTimeEnsemble, TradingStrategy};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch data from Bybit
//!     let client = BybitClient::new();
//!     let data = client.fetch_klines("BTCUSDT", "15", None, None, Some(1000)).await?;
//!
//!     // Create and train model
//!     let model = InceptionTimeEnsemble::new(5, 3)?;
//!
//!     // Generate trading signals
//!     let strategy = TradingStrategy::new(0.6, 0.02);
//!
//!     Ok(())
//! }
//! ```

pub mod backtest;
pub mod data;
pub mod model;
pub mod strategy;
pub mod training;
pub mod utils;

// Re-export main types for convenience
pub use data::{BybitClient, DataLoader, FeatureBuilder, NormalizationParams, OHLCVData, OHLCVDataset};
pub use model::{InceptionModule, InceptionTimeEnsemble, InceptionTimeNetwork};
pub use strategy::{PositionManager, RiskManager, SignalGenerator, TradingStrategy};
pub use training::{Trainer, TrainingConfig, TrainingMetrics};
pub use utils::{Config, setup_logging};
