//! Gated State Space Models for Trading
//!
//! This crate implements Gated SSMs (GSS-style) for financial time series
//! prediction and trading strategy generation.
//!
//! # Modules
//!
//! - `model`: Core SSM and Gated SSM implementations
//! - `data`: Data fetching (Bybit API) and feature engineering
//! - `backtest`: Backtesting engine with performance metrics
//! - `trading`: Signal generation and trading strategy

pub mod backtest;
pub mod data;
pub mod model;
pub mod trading;
