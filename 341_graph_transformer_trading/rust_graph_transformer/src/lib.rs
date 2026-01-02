//! # Graph Transformer Trading
//!
//! This library provides implementations for Graph Transformer models
//! applied to cryptocurrency trading using data from Bybit exchange.
//!
//! ## Core Concepts
//!
//! - **Market Graph**: Cryptocurrencies as nodes, correlations/relationships as edges
//! - **Graph Transformer**: Attention mechanism over graph structure
//! - **Trading Signals**: Predictions based on graph-aware embeddings
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `graph` - Graph construction and manipulation
//! - `transformer` - Graph Transformer layers and models
//! - `features` - Feature engineering from market data
//! - `strategy` - Trading signal generation
//! - `backtest` - Backtesting framework
//! - `utils` - Utility functions and helpers
//!
//! ## Example
//!
//! ```rust,no_run
//! use graph_transformer_trading::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch market data
//!     let client = BybitClient::new();
//!     let tickers = client.get_tickers().await?;
//!
//!     // Build market graph
//!     let graph_builder = MarketGraphBuilder::new();
//!     let market_graph = graph_builder.build(&tickers)?;
//!
//!     // Run Graph Transformer
//!     let model = GraphTransformer::new(GraphTransformerConfig::default());
//!     let embeddings = model.forward(&market_graph);
//!
//!     // Generate signals
//!     let signal_generator = SignalGenerator::new(0.6);
//!     let signals = signal_generator.generate(&embeddings, &market_graph);
//!
//!     for signal in signals {
//!         println!("{:?}", signal);
//!     }
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod backtest;
pub mod features;
pub mod graph;
pub mod strategy;
pub mod transformer;
pub mod utils;

// Re-export commonly used types
pub use api::client::BybitClient;
pub use api::types::{Kline, OrderBook, Ticker, Trade};
pub use api::websocket::BybitWebSocket;
pub use backtest::engine::BacktestEngine;
pub use backtest::report::BacktestReport;
pub use features::engine::FeatureEngine;
pub use features::indicators::TechnicalIndicators;
pub use graph::builder::MarketGraphBuilder;
pub use graph::market_graph::MarketGraph;
pub use graph::node::CryptoNode;
pub use strategy::signal::{Signal, SignalGenerator, SignalType};
pub use transformer::attention::GraphAttention;
pub use transformer::config::GraphTransformerConfig;
pub use transformer::model::GraphTransformer;
pub use transformer::positional::PositionalEncoding;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default trading symbols for examples
pub const DEFAULT_SYMBOLS: &[&str] = &[
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT",
];

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::api::client::BybitClient;
    pub use crate::api::types::{Kline, OrderBook, Ticker, Trade};
    pub use crate::api::websocket::BybitWebSocket;
    pub use crate::backtest::engine::BacktestEngine;
    pub use crate::backtest::report::BacktestReport;
    pub use crate::features::engine::FeatureEngine;
    pub use crate::features::indicators::TechnicalIndicators;
    pub use crate::graph::builder::MarketGraphBuilder;
    pub use crate::graph::market_graph::MarketGraph;
    pub use crate::graph::node::CryptoNode;
    pub use crate::strategy::signal::{Signal, SignalGenerator, SignalType};
    pub use crate::transformer::attention::GraphAttention;
    pub use crate::transformer::config::GraphTransformerConfig;
    pub use crate::transformer::model::GraphTransformer;
    pub use crate::transformer::positional::PositionalEncoding;
    pub use crate::DEFAULT_SYMBOLS;
}
