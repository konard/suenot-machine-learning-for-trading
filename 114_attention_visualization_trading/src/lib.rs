//! Attention Visualization for Trading
//!
//! This crate provides tools for visualizing and interpreting attention weights
//! in Transformer models applied to financial time series prediction.
//!
//! # Modules
//!
//! - `model`: Transformer implementation with attention extraction
//! - `data`: Data loading from Bybit and feature engineering
//! - `trading`: Signal generation and trading strategy
//! - `backtest`: Backtesting engine and performance metrics

pub mod model;
pub mod data;
pub mod trading;
pub mod backtest;

pub use model::{AttentionTransformer, AttentionWeights};
pub use data::{BybitClient, FeatureEngineering, Candle};
pub use trading::{Signal, TradingStrategy};
pub use backtest::{Backtester, TradingMetrics};
