//! # Equivariant GNN Trading Library
//!
//! A Rust implementation of E(n) Equivariant Graph Neural Networks
//! for cryptocurrency trading on Bybit exchange.
//!
//! ## Overview
//!
//! Equivariant GNNs respect the symmetries inherent in financial data:
//! - Permutation equivariance: Asset ordering doesn't affect predictions
//! - Scale invariance: Relative returns matter, not absolute prices
//! - Translation invariance: Feature differences drive predictions
//!
//! ## Architecture
//!
//! The library is organized into modular components:
//!
//! - `egnn`: Core E-GNN layer and network implementation
//! - `data`: Bybit API client and market data handling
//! - `features`: Technical indicators and feature extraction
//! - `trading`: Trading signals, backtesting, and risk management
//! - `utils`: Helper functions and common utilities
//!
//! ## Example
//!
//! ```rust,no_run
//! use equivariant_gnn_trading::{BybitClient, MarketGraph, EquivariantGNN};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch market data
//!     let client = BybitClient::new();
//!     let candles = client.get_klines("BTCUSDT", "60", 168, None, None).await?;
//!
//!     // Build market graph
//!     let graph_builder = MarketGraph::new(0.3);
//!     let graph = graph_builder.from_candles(&candles);
//!
//!     // Create E-GNN model
//!     let model = EquivariantGNN::new(10, 64, 3, 4);
//!
//!     // Generate predictions
//!     let signals = model.predict(&graph);
//!     println!("Trading signals: {:?}", signals);
//!
//!     Ok(())
//! }
//! ```

pub mod data;
pub mod egnn;
pub mod features;
pub mod trading;
pub mod utils;

// Re-export commonly used types
pub use data::{BybitClient, Candle, OrderBook, OrderBookLevel};
pub use egnn::{EGNNLayer, EquivariantGNN, MarketGraph, GraphNode, GraphEdge};
pub use features::{FeatureExtractor, TechnicalIndicators};
pub use trading::{TradingSignal, Position, Backtester, RiskManager, TradeDirection};
pub use utils::{normalize, standardize, correlation_matrix};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default hidden dimension for E-GNN layers
pub const DEFAULT_HIDDEN_DIM: usize = 64;

/// Default coordinate dimension for geometric embeddings
pub const DEFAULT_COORD_DIM: usize = 3;

/// Default number of E-GNN layers
pub const DEFAULT_NUM_LAYERS: usize = 4;

/// Default correlation threshold for graph construction
pub const DEFAULT_CORRELATION_THRESHOLD: f64 = 0.3;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_HIDDEN_DIM, 64);
        assert_eq!(DEFAULT_COORD_DIM, 3);
        assert_eq!(DEFAULT_NUM_LAYERS, 4);
        assert!((DEFAULT_CORRELATION_THRESHOLD - 0.3).abs() < 1e-10);
    }
}
