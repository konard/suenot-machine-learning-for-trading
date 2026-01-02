//! # Rust TCN Trading
//!
//! A modular Temporal Convolutional Network library for cryptocurrency trading on Bybit.
//!
//! ## Overview
//!
//! This library provides tools for building and using TCN models for financial
//! time series prediction and trading signal generation.
//!
//! ## Modules
//!
//! - `tcn` - TCN model implementation with causal dilated convolutions
//! - `features` - Feature engineering and technical indicators
//! - `trading` - Signal generation, risk management, and backtesting
//! - `api` - Bybit API client for fetching market data
//! - `utils` - Utility functions and performance metrics
//!
//! ## Example
//!
//! ```rust,no_run
//! use rust_tcn_trading::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch market data from Bybit
//!     let client = BybitClient::new();
//!     let data = client.get_klines("BTCUSDT", TimeFrame::Hour1, Some(1000), None, None).await?;
//!
//!     // Calculate features
//!     let features = TechnicalIndicators::calculate_all(&data.candles);
//!
//!     // Create and use TCN model
//!     let config = TCNConfig::default();
//!     let tcn = TCN::new(config);
//!
//!     // Generate trading signals
//!     let signal_gen = SignalGenerator::new(tcn, 0.6, 0.6);
//!     let signal = signal_gen.generate_signal(&features);
//!
//!     println!("Trading signal: {:?}", signal);
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod features;
pub mod tcn;
pub mod trading;
pub mod utils;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::api::{BybitClient, Candle, MarketData, TimeFrame};
    pub use crate::features::{FeatureMatrix, Normalizer, TechnicalIndicators};
    pub use crate::tcn::{TCNConfig, TCN};
    pub use crate::trading::{
        BacktestEngine, BacktestResult, RiskManager, SignalGenerator, SignalType, TradingSignal,
    };
    pub use crate::utils::{
        calculate_max_drawdown, calculate_profit_factor, calculate_sharpe_ratio,
        calculate_sortino_ratio, calculate_win_rate,
    };
}

// Re-export main types at crate root for convenience
pub use api::BybitClient;
pub use tcn::{TCNConfig, TCN};
pub use trading::SignalGenerator;
