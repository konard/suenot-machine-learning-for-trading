//! # MPNN Trading Library
//!
//! A Rust implementation of Message Passing Neural Networks for cryptocurrency trading.
//!
//! ## Features
//!
//! - **Graph Construction**: Build market graphs from correlation, sector, and order flow data
//! - **MPNN Layers**: Implement various message passing architectures (GCN, GAT, GraphSAGE)
//! - **Bybit Integration**: Fetch real-time and historical data from Bybit exchange
//! - **Trading Strategies**: Generate signals using graph-based analysis
//! - **Backtesting**: Evaluate strategy performance on historical data
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use mpnn_trading::{
//!     graph::MarketGraph,
//!     mpnn::MPNN,
//!     data::BybitClient,
//!     strategy::MPNNStrategy,
//! };
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Initialize Bybit client
//!     let client = BybitClient::new();
//!
//!     // Fetch market data
//!     let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
//!     let candles = client.fetch_candles(&symbols, "1h", 100).await?;
//!
//!     // Build market graph
//!     let graph = MarketGraph::from_correlations(&candles, 0.5)?;
//!
//!     // Create MPNN model
//!     let mpnn = MPNN::new(graph.node_count(), 64, 32, 3);
//!
//!     // Generate signals
//!     let signals = mpnn.forward(&graph)?;
//!
//!     Ok(())
//! }
//! ```

pub mod graph;
pub mod mpnn;
pub mod data;
pub mod strategy;
pub mod backtest;

// Re-export main types
pub use graph::{MarketGraph, Node, Edge};
pub use mpnn::{MPNN, MPNNConfig, AggregationType};
pub use data::{BybitClient, Candle, OrderBook};
pub use strategy::{MPNNStrategy, Signal, SignalType};
pub use backtest::{Backtester, BacktestResult, PerformanceMetrics};
