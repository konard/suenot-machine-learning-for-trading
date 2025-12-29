//! Data module for fetching and preprocessing cryptocurrency data
//!
//! This module provides:
//! - Bybit API client for fetching OHLCV data
//! - Data preprocessing and normalization
//! - DataLoader for batching sequences

mod bybit_client;
mod ohlcv;
mod loader;
mod preprocessing;

pub use bybit_client::BybitClient;
pub use ohlcv::{OHLCVData, OHLCVDataset};
pub use loader::DataLoader;
pub use preprocessing::{normalize_data, denormalize_data, create_sequences, NormalizationParams};
