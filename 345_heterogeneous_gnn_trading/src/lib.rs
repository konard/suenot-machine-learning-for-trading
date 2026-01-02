//! # Heterogeneous Graph Neural Networks for Trading
//!
//! This library implements Heterogeneous GNNs for cryptocurrency trading on Bybit.
//! Unlike homogeneous GNNs, this implementation supports multiple node types
//! (assets, exchanges, wallets) and edge types (correlation, trades_on, holds).
//!
//! ## Modules
//!
//! - `graph` - Heterogeneous graph data structures with typed nodes and edges
//! - `gnn` - Graph neural network layers with type-aware attention
//! - `data` - Bybit API client and data processing
//! - `strategy` - Trading signal generation and execution
//! - `utils` - Utilities and metrics
//!
//! ## Example
//!
//! ```rust,no_run
//! use heterogeneous_gnn_trading::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a heterogeneous graph for crypto market
//!     let schema = GraphSchema::trading_schema();
//!     let mut graph = HeterogeneousGraph::new(schema);
//!
//!     // Add asset nodes
//!     let btc_features = AssetFeatures::new(50000.0, 1_000_000.0, 0.02);
//!     graph.add_node("BTCUSDT", NodeType::Asset, btc_features.into());
//!
//!     // Add exchange nodes
//!     let bybit_features = ExchangeFeatures::new(5_000_000.0, 100, 0.95);
//!     graph.add_node("Bybit", NodeType::Exchange, bybit_features.into());
//!
//!     // Add typed edge
//!     graph.add_edge("BTCUSDT", "Bybit", EdgeType::TradesOn, EdgeFeatures::default());
//!
//!     // Create HGNN model
//!     let config = HGNNConfig::default();
//!     let model = HeterogeneousGNN::new(config);
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
    pub use crate::graph::{
        HeterogeneousGraph, GraphSchema, GraphSnapshot,
        Node, NodeType, NodeId, NodeFeatures,
        Edge, EdgeType, EdgeId, EdgeFeatures,
        AssetFeatures, ExchangeFeatures, WalletFeatures, MarketConditionFeatures,
        Metapath, MetapathInstance,
    };
    pub use crate::gnn::{
        HeterogeneousGNN, HGNNConfig, HGNNLayer,
        TypeProjection, RelationAttention, SemanticAggregation,
    };
    pub use crate::data::{
        BybitClient, BybitConfig, BybitError,
        Kline, OrderBook, Ticker, Trade, FundingRate, OpenInterest,
    };
    pub use crate::strategy::{
        Signal, SignalType, TradingStrategy, StrategyConfig,
        MetapathSignal, SignalAggregator,
    };
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
