//! # Concept Bottleneck Trading
//!
//! A Rust implementation of Concept Bottleneck Models for interpretable algorithmic trading.
//!
//! ## Overview
//!
//! This library provides tools for building trading systems that make decisions through
//! human-interpretable intermediate concepts, allowing for:
//!
//! - **Interpretability**: Understanding why trading decisions are made
//! - **Intervention**: Overriding concept predictions at inference time
//! - **Debugging**: Identifying failure modes in the trading logic
//!
//! ## Modules
//!
//! - `api` - API clients for fetching market data (Bybit)
//! - `data` - Data processing and feature engineering
//! - `models` - Concept Bottleneck Model implementations
//! - `utils` - Technical indicators and utility functions
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use concept_bottleneck_trading::api::BybitClient;
//! use concept_bottleneck_trading::models::ConceptBottleneckTrader;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Fetch market data
//!     let client = BybitClient::new();
//!     let klines = client.fetch_klines("BTCUSDT", "60", 200).await?;
//!
//!     // Create model and predict
//!     let model = ConceptBottleneckTrader::new();
//!     let (signal, concepts) = model.predict(&klines);
//!
//!     println!("Signal: {:?}", signal);
//!     println!("Concepts: {:?}", concepts);
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod data;
pub mod models;
pub mod utils;

// Re-export main types
pub use api::bybit::BybitClient;
pub use data::processor::DataProcessor;
pub use models::cbm::ConceptBottleneckTrader;
pub use models::concepts::{TradingConcepts, TrendDirection, VolatilityRegime, MomentumSignal};
