//! # Graph Neural Networks for Cryptocurrency Trading
//!
//! This library implements Graph Neural Networks (GNN) for analyzing
//! cryptocurrency relationships and building trading strategies based
//! on momentum propagation using data from Bybit exchange.
//!
//! ## Features
//!
//! - Bybit API client for fetching historical OHLCV data
//! - Graph construction from correlation matrices, k-NN, and sectors
//! - GCN and GAT model implementations
//! - Lead-lag detection and momentum propagation strategies
//! - Backtesting framework
//!
//! ## Example
//!
//! ```rust,no_run
//! use gnn_crypto::{
//!     data::BybitClient,
//!     graph::CorrelationGraph,
//!     model::{GCN, GNNConfig},
//!     strategy::MomentumStrategy,
//! };
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch data for multiple cryptocurrencies
//!     let client = BybitClient::new();
//!     let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
//!     let data = client.fetch_multi_symbol(&symbols, "60", 90).await?;
//!
//!     // Build correlation graph
//!     let graph_builder = CorrelationGraph::new(0.5, 60);
//!     let crypto_graph = graph_builder.build(&data)?;
//!
//!     // Create GNN model
//!     let config = GNNConfig::default();
//!     let model = GCN::new(&config, tch::Device::Cpu);
//!
//!     Ok(())
//! }
//! ```

pub mod data;
pub mod graph;
pub mod model;
pub mod strategy;
pub mod utils;

// Re-export main types
pub use data::{BybitClient, FeatureEngineer, OHLCV};
pub use graph::{CorrelationGraph, CryptoGraph, KNNGraph};
pub use model::{GCN, GAT, GNNConfig};
pub use strategy::{MomentumStrategy, Signal, TradingStrategy};
pub use utils::Config;
