//! # Graph Pooling for Trading
//!
//! A Rust implementation of hierarchical graph pooling for cryptocurrency trading.
//! Uses Bybit API for real-time market data.
//!
//! ## Features
//!
//! - **Graph Construction**: Build correlation graphs from price data
//! - **Graph Pooling**: DiffPool, SAGPool, MinCutPool implementations
//! - **Trading Strategy**: Cluster-relative momentum with regime detection
//! - **Bybit Integration**: Real-time data from Bybit cryptocurrency exchange
//!
//! ## Example
//!
//! ```rust,no_run
//! use graph_pooling_trading::{BybitClient, MarketGraph, HierarchicalPooling};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Initialize Bybit client
//!     let client = BybitClient::new(None, None);
//!
//!     // Fetch market data
//!     let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
//!     let klines = client.get_klines_batch(&symbols, "1h", 100).await?;
//!
//!     // Build market graph
//!     let graph = MarketGraph::from_klines(&klines)?;
//!
//!     // Apply hierarchical pooling
//!     let pooling = HierarchicalPooling::new(10, 3);
//!     let result = pooling.forward(&graph)?;
//!
//!     println!("Discovered {} clusters", result.cluster_assignments.ncols());
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod graph;
pub mod pooling;
pub mod strategy;

// Re-exports for convenience
pub use api::{BybitClient, Kline, OrderBook, Ticker};
pub use graph::{MarketGraph, CorrelationGraph};
pub use pooling::{DiffPool, SAGPool, MinCutPool, HierarchicalPooling, PoolingResult};
pub use strategy::{GraphPoolingStrategy, Signal, Regime, StrategyConfig};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default symbols to track
pub const DEFAULT_SYMBOLS: &[&str] = &[
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT", "LTCUSDT",
    "SHIBUSDT", "AVAXUSDT", "LINKUSDT", "ATOMUSDT", "UNIUSDT",
    "ETCUSDT", "XLMUSDT", "NEARUSDT", "ALGOUSDT", "AAVEUSDT",
];

/// Prelude module for common imports
pub mod prelude {
    pub use crate::api::{BybitClient, Kline};
    pub use crate::graph::MarketGraph;
    pub use crate::pooling::{HierarchicalPooling, PoolingResult};
    pub use crate::strategy::{GraphPoolingStrategy, Signal};
}
