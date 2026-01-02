//! DenseNet Architecture Implementation
//!
//! This module implements the DenseNet (Densely Connected Convolutional Networks)
//! architecture for time series prediction in cryptocurrency trading.

mod layer;
mod block;
mod transition;
mod network;
mod config;

pub use layer::DenseLayer;
pub use block::DenseBlock;
pub use transition::TransitionLayer;
pub use network::DenseNet;
pub use config::DenseNetConfig;
