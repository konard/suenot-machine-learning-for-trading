//! # SHAP Trading Interpretability
//!
//! This crate implements SHAP (SHapley Additive exPlanations) for explaining
//! trading model predictions. SHAP provides a unified measure of feature importance
//! based on game-theoretic Shapley values.
//!
//! ## Features
//!
//! - SHAP value computation for trading models
//! - KernelSHAP approximation for model-agnostic explanations
//! - Bybit API integration for cryptocurrency data
//! - Trading strategy with SHAP-based confidence filtering
//! - Backtesting framework for strategy evaluation
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use shap_trading::{ShapExplainer, TradingStrategy, BybitClient};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let client = BybitClient::new();
//!     let data = client.fetch_klines("BTCUSDT", "60", 1000).await?;
//!
//!     let explainer = ShapExplainer::new(feature_names, base_value);
//!     let explanation = explainer.explain(&model, &features);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Reference
//!
//! Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions"
//! NeurIPS 2017. arXiv:1705.07874

pub mod model;
pub mod data;
pub mod trading;
pub mod backtest;

pub use model::shap::{ShapExplainer, ShapExplanation, ShapValues};
pub use data::bybit::BybitClient;
pub use trading::strategy::TradingStrategy;
pub use trading::signals::TradingSignal;
pub use backtest::engine::BacktestEngine;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::model::shap::{ShapExplainer, ShapExplanation, ShapValues};
    pub use crate::data::bybit::BybitClient;
    pub use crate::trading::strategy::TradingStrategy;
    pub use crate::trading::signals::TradingSignal;
    pub use crate::backtest::engine::BacktestEngine;
}

/// Error types for the crate.
#[derive(thiserror::Error, Debug)]
pub enum ShapError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("JSON parsing failed: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Insufficient data: need {needed}, got {got}")]
    InsufficientData { needed: usize, got: usize },

    #[error("Computation error: {0}")]
    ComputationError(String),
}

pub type Result<T> = std::result::Result<T, ShapError>;
