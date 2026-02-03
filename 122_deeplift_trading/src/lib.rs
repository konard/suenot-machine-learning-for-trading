//! DeepLIFT Trading Library
//!
//! This library provides DeepLIFT attribution for neural network trading models,
//! enabling interpretable trading decisions with Bybit cryptocurrency data.
//!
//! # Modules
//!
//! - `deeplift`: Core DeepLIFT attribution implementation
//! - `model`: Neural network models for trading
//! - `data`: Data loading and feature engineering
//! - `trading`: Trading strategy components
//! - `backtest`: Backtesting framework with explanations

pub mod backtest;
pub mod data;
pub mod deeplift;
pub mod model;
pub mod trading;

pub use backtest::engine::BacktestEngine;
pub use data::bybit::BybitClient;
pub use data::features::FeatureGenerator;
pub use deeplift::attribution::{Attribution, DeepLIFT};
pub use model::network::TradingNetwork;
pub use trading::signals::TradingSignal;
pub use trading::strategy::TradingStrategy;
