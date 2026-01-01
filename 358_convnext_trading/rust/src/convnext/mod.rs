//! ConvNeXt architecture module
//!
//! Implementation of ConvNeXt for 1D time series processing.

mod block;
mod layers;
mod model;

pub use block::ConvNeXtBlock;
pub use layers::{Conv1d, LayerNorm, Linear, DropPath};
pub use model::{ConvNeXt, ConvNeXtConfig};
