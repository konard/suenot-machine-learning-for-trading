//! Equivariant Graph Neural Network Module
//!
//! Implementation of E(n) Equivariant Graph Neural Networks for trading.
//! The architecture preserves symmetries in financial data including:
//! - Permutation equivariance (asset ordering invariance)
//! - Translation equivariance (feature differences matter, not absolutes)
//! - Scale invariance (relative movements, not absolute prices)

mod layer;
mod network;
mod graph;
mod config;

pub use layer::EGNNLayer;
pub use network::EquivariantGNN;
pub use graph::{MarketGraph, GraphNode, GraphEdge, Graph};
pub use config::EGNNConfig;
