//! Graph module for dynamic graph data structures
//!
//! This module provides the core data structures for representing
//! dynamic graphs that evolve over time.

mod node;
mod edge;
mod dynamic;

pub use node::{Node, NodeFeatures, NodeId};
pub use edge::{Edge, EdgeFeatures, EdgeId};
pub use dynamic::{DynamicGraph, GraphSnapshot, TemporalEdge};

/// Graph configuration
#[derive(Debug, Clone)]
pub struct GraphConfig {
    /// Maximum number of nodes
    pub max_nodes: usize,
    /// Correlation threshold for creating edges
    pub correlation_threshold: f64,
    /// Time window for rolling correlation (in seconds)
    pub correlation_window: u64,
    /// Enable temporal edges
    pub temporal_edges: bool,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            max_nodes: 100,
            correlation_threshold: 0.5,
            correlation_window: 3600, // 1 hour
            temporal_edges: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_config_default() {
        let config = GraphConfig::default();
        assert_eq!(config.max_nodes, 100);
        assert_eq!(config.correlation_threshold, 0.5);
    }
}
