//! Synthetic Control Method for Trading
//!
//! This crate provides tools for applying the Synthetic Control Method (SCM)
//! to financial event studies and trading strategies.
//!
//! # Overview
//!
//! The Synthetic Control Method constructs a "synthetic" version of a treated
//! unit (e.g., a stock affected by an event) using a weighted combination of
//! control units (similar stocks unaffected by the event).
//!
//! # Modules
//!
//! - `model`: Core SCM implementation with weight optimization
//! - `data`: Data fetching from Bybit and other sources
//! - `backtest`: Backtesting framework for event-driven strategies
//! - `trading`: Signal generation and trading utilities
//!
//! # Example
//!
//! ```rust,ignore
//! use synthetic_control_trading::{SyntheticControlModel, BybitClient};
//! use synthetic_control_trading::data::bybit::extract_returns;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch data
//!     let client = BybitClient::new();
//!     let treated_klines = client.fetch_klines("BTCUSDT", "D", 100).await?;
//!     let donor_klines = client.fetch_klines("ETHUSDT", "D", 100).await?;
//!
//!     // Extract returns
//!     let treated = extract_returns(&treated_klines);
//!     let donors = vec![extract_returns(&donor_klines)];
//!
//!     // Fit model
//!     let mut model = SyntheticControlModel::new();
//!     model.fit(&treated[..60], &donors)?;
//!
//!     Ok(())
//! }
//! ```

pub mod model;
pub mod data;
pub mod backtest;
pub mod trading;

// Re-exports for convenience
pub use model::synthetic_control::SyntheticControlModel;
pub use data::bybit::BybitClient;
pub use backtest::engine::BacktestEngine;
pub use trading::signals::SignalGenerator;
