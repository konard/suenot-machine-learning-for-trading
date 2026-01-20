//! # LLM Anomaly Detection
//!
//! This crate provides tools for detecting anomalies in financial data
//! using statistical methods and LLM-based approaches.
//!
//! ## Features
//!
//! - Statistical anomaly detection (Z-score, Mahalanobis distance)
//! - Data loading from Yahoo Finance and Bybit
//! - Trading signal generation
//! - Backtesting framework
//!
//! ## Example
//!
//! ```rust,no_run
//! use llm_anomaly_detection::{
//!     data_loader::BybitLoader,
//!     detector::StatisticalDetector,
//! };
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Load data from Bybit
//!     let loader = BybitLoader::new();
//!     let candles = loader.get_klines("BTCUSDT", "1h", 500).await?;
//!
//!     // Create and train detector
//!     let mut detector = StatisticalDetector::new(2.5);
//!     detector.fit(&candles)?;
//!
//!     // Detect anomalies
//!     let result = detector.detect(&candles.last().unwrap())?;
//!     println!("Is anomaly: {}", result.is_anomaly);
//!
//!     Ok(())
//! }
//! ```

pub mod data_loader;
pub mod detector;
pub mod signals;
pub mod backtest;
pub mod types;

pub use types::*;
