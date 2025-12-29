//! # PCA Crypto - Principal Component Analysis for Cryptocurrency Trading
//!
//! This library provides tools for applying Principal Component Analysis (PCA)
//! to cryptocurrency market data from Bybit exchange.
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `data` - Data structures and preprocessing utilities
//! - `pca` - PCA implementation and analysis
//! - `portfolio` - Eigenportfolio construction and analysis
//! - `utils` - Helper functions and statistics

pub mod api;
pub mod data;
pub mod pca;
pub mod portfolio;
pub mod utils;

pub use api::BybitClient;
pub use data::{MarketData, Returns};
pub use pca::PCAAnalysis;
pub use portfolio::Eigenportfolio;
