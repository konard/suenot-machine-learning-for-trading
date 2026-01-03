//! GLOW Model implementation
//!
//! This module provides:
//! - ActNorm: Data-dependent activation normalization
//! - InvertibleConv1x1: Invertible 1x1 convolution with LU decomposition
//! - AffineCoupling: Affine coupling layer
//! - FlowStep: Complete flow step (ActNorm + 1x1 Conv + Coupling)
//! - GLOWModel: Full GLOW model with multi-scale architecture

mod layers;
mod glow;

pub use layers::{ActNorm, InvertibleConv1x1, AffineCoupling, FlowStep};
pub use glow::{GLOWModel, GLOWConfig};
