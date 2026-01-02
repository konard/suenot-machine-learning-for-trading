//! Graph data structures and builders
//!
//! Provides structures for representing asset relationship graphs
//! and methods to construct them from market data.

mod builder;
mod sparse;

pub use builder::GraphBuilder;
pub use sparse::SparseGraph;
