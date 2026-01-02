//! # Graph Generation for Trading
//!
//! A comprehensive library for building and analyzing graph-based representations
//! of cryptocurrency markets using data from Bybit exchange.
//!
//! ## Features
//!
//! - **Data Module**: Bybit API integration and data preprocessing
//! - **Graph Module**: Various graph construction methods (correlation, visibility, etc.)
//! - **Models Module**: Graph neural network implementations
//! - **Trading Module**: Signal generation and backtesting
//! - **Utils Module**: Mathematical utilities
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use graph_generation_trading::{
//!     data::BybitClient,
//!     graph::{GraphBuilder, CorrelationMethod},
//!     trading::GraphSignals,
//! };
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch market data
//!     let client = BybitClient::new();
//!     let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
//!     let data = client.fetch_klines(&symbols, "1h", 100).await?;
//!
//!     // Build correlation graph
//!     let graph = GraphBuilder::new()
//!         .with_method(CorrelationMethod::Pearson)
//!         .with_threshold(0.7)
//!         .build(&data)?;
//!
//!     // Generate trading signals
//!     let signals = GraphSignals::new(&graph);
//!     let centrality = signals.betweenness_centrality();
//!
//!     Ok(())
//! }
//! ```

pub mod data;
pub mod graph;
pub mod models;
pub mod trading;
pub mod utils;

// Re-export commonly used types
pub use data::{BybitClient, MarketData, OHLCV};
pub use graph::{CorrelationMethod, GraphBuilder, MarketGraph};
pub use trading::{BacktestEngine, GraphSignals, Portfolio};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
