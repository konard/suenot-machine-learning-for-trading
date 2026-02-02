//! PCMCI Causal Discovery for Trading
//!
//! This crate implements the PCMCI (Peter and Clark Momentary Conditional Independence)
//! algorithm for causal discovery in financial time series data.
//!
//! # Features
//!
//! - PCMCI algorithm with PC-stable phase and MCI testing
//! - Partial correlation and conditional mutual information tests
//! - Causal graph construction with significance testing
//! - Data loaders for yfinance CSV and Bybit API
//! - Causality-driven trading strategy and backtesting
//!
//! # Example
//!
//! ```rust,no_run
//! use pcmci_causal_discovery::{
//!     PCMCICausalDiscovery, PCMCIConfig, IndependenceTest,
//! };
//!
//! // Create configuration
//! let config = PCMCIConfig::default();
//!
//! // Initialize PCMCI
//! let pcmci = PCMCICausalDiscovery::new(config);
//! ```

pub mod model;
pub mod data;
pub mod strategy;

// Re-exports
pub use model::{PCMCICausalDiscovery, PCMCIConfig, IndependenceTest, CausalLink, CausalGraph};
pub use data::{OHLCVBar, MarketData, load_csv_data, BybitClient};
pub use strategy::{BacktestConfig, CausalTradingStrategy, Backtester, BacktestResult};
