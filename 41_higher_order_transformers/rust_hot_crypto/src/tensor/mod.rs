//! Tensor operations and decomposition
//!
//! This module provides:
//! - 3D tensor type and operations
//! - CP (CANDECOMP/PARAFAC) decomposition
//! - Tucker decomposition (optional)

mod tensor3d;
mod decomposition;
mod operations;

pub use tensor3d::Tensor3D;
pub use decomposition::{CPDecomposition, CPResult};
pub use operations::*;
