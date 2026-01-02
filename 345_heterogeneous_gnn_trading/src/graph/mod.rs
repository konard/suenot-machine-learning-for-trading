//! Heterogeneous graph module for typed nodes and edges
//!
//! This module provides data structures for representing heterogeneous graphs
//! where nodes and edges can have different types.

mod node;
mod edge;
mod schema;
mod heterogeneous;

pub use node::{Node, NodeType, NodeId, NodeFeatures, AssetFeatures, ExchangeFeatures, WalletFeatures, MarketConditionFeatures};
pub use edge::{Edge, EdgeType, EdgeId, EdgeFeatures};
pub use schema::{GraphSchema, Metapath, MetapathInstance};
pub use heterogeneous::{HeterogeneousGraph, GraphSnapshot};

/// Graph configuration for heterogeneous graphs
#[derive(Debug, Clone)]
pub struct GraphConfig {
    /// Maximum number of nodes per type
    pub max_nodes_per_type: usize,
    /// Correlation threshold for creating correlation edges
    pub correlation_threshold: f64,
    /// Time window for rolling correlation (in seconds)
    pub correlation_window: u64,
    /// Enable metapath sampling
    pub enable_metapaths: bool,
    /// Maximum metapath length
    pub max_metapath_length: usize,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            max_nodes_per_type: 100,
            correlation_threshold: 0.5,
            correlation_window: 3600, // 1 hour
            enable_metapaths: true,
            max_metapath_length: 5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_config_default() {
        let config = GraphConfig::default();
        assert_eq!(config.max_nodes_per_type, 100);
        assert_eq!(config.correlation_threshold, 0.5);
        assert!(config.enable_metapaths);
    }
}
