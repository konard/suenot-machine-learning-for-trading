//! Feature engineering module for anomaly detection
//!
//! Provides functions to compute features from OHLCV data that are
//! useful for detecting anomalies in cryptocurrency markets.

mod indicators;
mod engine;

pub use indicators::*;
pub use engine::*;
