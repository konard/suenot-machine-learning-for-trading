//! # Subgraph Mining for Trading
//!
//! A Rust library for discovering and trading on subgraph patterns
//! in cryptocurrency markets.
//!
//! ## Overview
//!
//! This library provides:
//! - Graph construction from market data (correlation networks, order flow)
//! - Subgraph pattern mining using gSpan algorithm
//! - Pattern matching for trading signal generation
//! - Backtesting framework for strategy validation
//! - Bybit API integration for real-time data
//!
//! ## Example
//!
//! ```rust,no_run
//! use subgraph_mining_trading::{
//!     data::BybitClient,
//!     graph::GraphBuilder,
//!     mining::GSpanMiner,
//!     trading::SignalGenerator,
//! };
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch market data
//!     let client = BybitClient::new();
//!     let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
//!     let candles = client.fetch_candles(&symbols, "1h", 100).await?;
//!
//!     // Build correlation graph
//!     let builder = GraphBuilder::new()
//!         .correlation_threshold(0.7)
//!         .window_size(24);
//!     let graph = builder.build_correlation_network(&candles)?;
//!
//!     // Mine frequent subgraph patterns
//!     let miner = GSpanMiner::new().min_support(3);
//!     let patterns = miner.mine(&graph)?;
//!
//!     // Generate trading signals
//!     let signal_gen = SignalGenerator::new(patterns);
//!     let signals = signal_gen.generate(&graph)?;
//!
//!     Ok(())
//! }
//! ```

pub mod data;
pub mod graph;
pub mod mining;
pub mod trading;
pub mod utils;

// Re-export main types for convenience
pub use data::{BybitClient, Candle, MarketData};
pub use graph::{FinancialGraph, GraphBuilder, Node, Edge};
pub use mining::{GSpanMiner, Pattern, PatternMatch};
pub use trading::{Signal, SignalGenerator, Strategy, BacktestResult};
pub use utils::{Config, Stats};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::data::{BybitClient, Candle, MarketData};
    pub use crate::graph::{FinancialGraph, GraphBuilder, Node, Edge};
    pub use crate::mining::{GSpanMiner, Pattern, PatternMatch};
    pub use crate::trading::{Signal, SignalGenerator, Strategy, BacktestResult};
    pub use crate::utils::{Config, Stats};
}
