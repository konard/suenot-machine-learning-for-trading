//! Data module for fetching, preprocessing, and managing trading data
//!
//! This module provides:
//! - Bybit API client for fetching OHLCV data
//! - OHLCV data structures
//! - Feature engineering for technical indicators
//! - Dataset management and loading

mod bybit_client;
mod dataset;
mod features;
mod ohlcv;

pub use bybit_client::BybitClient;
pub use dataset::{DataLoader, TradingDataset};
pub use features::{FeatureBuilder, NormalizationParams};
pub use ohlcv::{OHLCVData, OHLCVDataset};
