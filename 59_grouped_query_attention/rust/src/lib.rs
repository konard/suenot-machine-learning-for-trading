//! # GQA Trading - Grouped Query Attention for Trading
//!
//! This crate implements Grouped Query Attention (GQA) for financial
//! time series prediction, including cryptocurrency and stock trading.
//!
//! ## Key Features
//!
//! - Efficient GQA implementation with reduced KV cache memory
//! - Data loading from Bybit and Yahoo Finance
//! - Backtesting framework for strategy evaluation
//!
//! ## Example
//!
//! ```rust,no_run
//! use gqa_trading::{GQATrader, load_bybit_data};
//!
//! // Load data
//! let data = load_bybit_data("BTCUSDT", "1h", 1000).unwrap();
//!
//! // Create model
//! let model = GQATrader::new(5, 64, 8, 2, 4);
//!
//! // Make prediction
//! let sequence = data.slice(s![..60, ..]).to_owned();
//! let prediction = model.predict(&sequence);
//! ```

pub mod model;
pub mod data;
pub mod strategy;
pub mod predict;

// Re-exports for convenience
pub use model::{GroupedQueryAttention, GQATrader, GQABlock};
pub use data::{load_bybit_data, load_yahoo_data, OHLCVData, normalize_data};
pub use strategy::{backtest_strategy, BacktestResult, Trade};
pub use predict::{predict_next, PredictionResult};
