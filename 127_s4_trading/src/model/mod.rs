//! S4 model implementations.
//!
//! This module provides structured state space models for sequence processing:
//! - S4Layer: Full S4 with HiPPO initialization
//! - S4DLayer: Simplified diagonal S4
//! - S4Model: Complete model with multiple S4 layers

pub mod s4;

pub use s4::{S4Layer, S4Model, S4Config, S4DLayer};
