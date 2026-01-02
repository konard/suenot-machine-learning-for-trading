//! # Dynamic Graph Neural Networks for Trading
//!
//! This library implements Dynamic GNNs for cryptocurrency trading on Bybit.
//!
//! ## Modules
//!
//! - `graph` - Dynamic graph data structures and operations
//! - `gnn` - Graph neural network layers and models
//! - `data` - Bybit API client and data processing
//! - `strategy` - Trading signal generation and execution
//! - `utils` - Utilities and metrics
//!
//! ## Example
//!
//! ```rust,no_run
//! use dynamic_gnn_trading::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a dynamic graph for crypto assets
//!     let mut graph = DynamicGraph::new();
//!
//!     // Add nodes for different cryptocurrencies
//!     graph.add_node("BTCUSDT", NodeFeatures::default());
//!     graph.add_node("ETHUSDT", NodeFeatures::default());
//!
//!     // Create GNN model
//!     let config = GNNConfig::default();
//!     let model = DynamicGNN::new(config);
//!
//!     Ok(())
//! }
//! ```

pub mod graph;
pub mod gnn;
pub mod data;
pub mod strategy;
pub mod utils;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::graph::{DynamicGraph, Node, Edge, NodeFeatures, EdgeFeatures};
    pub use crate::gnn::{DynamicGNN, GNNConfig, GNNLayer};
    pub use crate::data::{BybitClient, BybitConfig, Kline, OrderBook, Ticker};
    pub use crate::strategy::{Signal, SignalType, TradingStrategy, StrategyConfig};
    pub use crate::utils::{Metrics, PerformanceTracker};
}

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
