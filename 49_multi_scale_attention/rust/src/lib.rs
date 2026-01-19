//! Multi-Scale Attention for Financial Time Series
//!
//! This crate provides a Rust implementation of multi-scale attention
//! mechanisms for financial time series prediction. It captures patterns
//! across different temporal resolutions simultaneously.

pub mod model;
pub mod data;
pub mod strategy;

pub use model::{
    MultiScaleAttention,
    MultiScaleAttentionConfig,
    ScaleEncoder,
    ScaleConfig,
};
pub use data::{
    fetch_bybit_klines,
    create_multi_scale_features,
    create_sequences,
    generate_synthetic_data,
    BybitKline,
    MultiScaleData,
};
pub use strategy::{
    backtest_multi_scale_strategy,
    BacktestResult,
    BacktestConfig,
};
