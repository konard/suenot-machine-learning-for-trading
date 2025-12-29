//! # Rust Neural Network for Cryptocurrency Trading
//!
//! This library provides a modular implementation of feedforward neural networks
//! for cryptocurrency trading using Bybit exchange data.
//!
//! ## Modules
//!
//! - `nn` - Neural network implementation (layers, activations, training)
//! - `data` - Data fetching and preprocessing from Bybit API
//! - `features` - Technical indicators and feature engineering
//! - `strategy` - Trading strategies based on NN predictions
//! - `backtest` - Backtesting engine for strategy evaluation

pub mod nn;
pub mod data;
pub mod features;
pub mod strategy;
pub mod backtest;

pub use nn::NeuralNetwork;
pub use data::BybitClient;
pub use features::FeatureEngine;
pub use strategy::TradingStrategy;
pub use backtest::Backtester;
