//! # Data Module
//!
//! This module handles data fetching from Bybit exchange and preprocessing
//! for use with Neural ODE models.
//!
//! ## Components
//!
//! - [`bybit`]: Bybit API client for fetching market data
//! - [`candles`]: Candlestick data structures and utilities
//! - [`features`]: Technical indicators and feature engineering

mod bybit;
mod candles;
mod features;

pub use bybit::BybitClient;
pub use candles::{Candle, CandleData, Timeframe};
pub use features::{Features, TechnicalIndicators, Symbol};
