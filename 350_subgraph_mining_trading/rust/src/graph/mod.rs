//! Graph module for financial network construction and manipulation.
//!
//! This module provides:
//! - Core graph types (Node, Edge, FinancialGraph)
//! - Graph construction from market data
//! - Graph algorithms for analysis

mod types;
mod builder;
mod algorithms;

pub use types::{Node, Edge, NodeId, EdgeWeight, FinancialGraph, GraphType};
pub use builder::GraphBuilder;
pub use algorithms::{find_triangles, find_cliques, find_stars, compute_centrality};
