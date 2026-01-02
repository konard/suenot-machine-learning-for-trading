//! Graph Attention Network implementation
//!
//! Provides GAT layers and networks for processing graph-structured data.

mod attention;
mod layer;
mod network;

pub use attention::{AttentionHead, MultiHeadAttention};
pub use layer::GraphAttentionLayer;
pub use network::GraphAttentionNetwork;
