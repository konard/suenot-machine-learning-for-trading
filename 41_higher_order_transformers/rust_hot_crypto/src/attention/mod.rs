//! Attention mechanisms for transformers
//!
//! This module provides:
//! - Standard scaled dot-product attention
//! - Higher-order attention with CP decomposition
//! - Kernel attention for linear complexity

mod standard;
mod higher_order;
mod kernel;

pub use standard::StandardAttention;
pub use higher_order::HigherOrderAttention;
pub use kernel::KernelAttention;
