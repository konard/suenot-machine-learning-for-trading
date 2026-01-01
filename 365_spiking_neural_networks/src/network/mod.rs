//! Network architecture for Spiking Neural Networks
//!
//! This module provides layer-based network construction and topology management.

mod layer;
mod topology;

pub use layer::{SNNLayer, LayerConfig, LayerType};
pub use topology::{SNNNetwork, NetworkConfig};
