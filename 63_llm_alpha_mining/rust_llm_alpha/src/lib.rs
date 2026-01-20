//! LLM Alpha Mining - Rust Implementation
//!
//! A high-performance toolkit for LLM-powered alpha factor mining
//! in cryptocurrency trading.
//!
//! # Overview
//!
//! This library provides:
//! - Data loading from Bybit and other sources
//! - Alpha factor generation and evaluation
//! - Backtesting framework
//! - Self-improving QuantAgent
//!
//! # Example
//!
//! ```rust,ignore
//! use llm_alpha_mining::{data::BybitLoader, alpha::AlphaEvaluator};
//!
//! #[tokio::main]
//! async fn main() {
//!     let loader = BybitLoader::new();
//!     let data = loader.load("BTCUSDT", "60", 30).await.unwrap();
//!     println!("Loaded {} candles", data.len());
//! }
//! ```

pub mod data;
pub mod alpha;
pub mod backtest;
pub mod quantagent;
pub mod error;

pub use error::{Error, Result};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
