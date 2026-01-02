//! Graph Attention Networks for Cryptocurrency Trading
//!
//! This crate provides a complete implementation of Graph Attention Networks (GAT)
//! for algorithmic trading on cryptocurrency markets using Bybit exchange data.
//!
//! # Features
//!
//! - **Graph Construction**: Build asset relationship graphs from correlation, sector, or k-NN
//! - **GAT Implementation**: Multi-head attention with sparse operations
//! - **Feature Engineering**: Technical indicators and cross-asset features
//! - **Trading Signals**: Signal generation and propagation through the network
//! - **Backtesting**: Performance evaluation with standard metrics
//!
//! # Example
//!
//! ```rust,no_run
//! use gat_trading::{api::BybitClient, gat::GraphAttentionNetwork, graph::GraphBuilder};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch market data
//!     let client = BybitClient::new();
//!     let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
//!
//!     // Build asset graph
//!     let graph = GraphBuilder::from_correlation(&returns, 0.5)?;
//!
//!     // Create and run GAT
//!     let gat = GraphAttentionNetwork::new(64, 32, 4)?;
//!     let embeddings = gat.forward(&features, &graph);
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod backtest;
pub mod features;
pub mod gat;
pub mod graph;
pub mod trading;

// Re-export commonly used types
pub use api::BybitClient;
pub use backtest::Backtester;
pub use gat::{GraphAttentionLayer, GraphAttentionNetwork};
pub use graph::{GraphBuilder, SparseGraph};
pub use trading::{SignalGenerator, TradingStrategy};
