//! TCN (Temporal Convolutional Network) implementation
//!
//! This module provides a complete TCN implementation with:
//! - Causal dilated convolutions
//! - Residual blocks
//! - Configurable architecture

mod block;
mod layer;
mod model;

pub use block::TCNResidualBlock;
pub use layer::CausalConv1d;
pub use model::{TCNConfig, TCN};
