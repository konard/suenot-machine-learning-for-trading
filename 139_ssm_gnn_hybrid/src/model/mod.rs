//! Core SSM-GNN hybrid model components.

mod ssm;
mod gnn;
mod hybrid;

pub use ssm::DiagonalSSM;
pub use gnn::GraphAttentionLayer;
pub use hybrid::{SsmGnnHybrid, Signal, Prediction};
