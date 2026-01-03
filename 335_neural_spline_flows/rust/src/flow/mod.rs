//! Flow module for Neural Spline Flows implementation
//!
//! This module contains the core components of Neural Spline Flows:
//! - Rational-quadratic spline transformations
//! - Coupling layers
//! - Complete NSF model

pub mod coupling;
pub mod nsf;
pub mod spline;

pub use coupling::CouplingLayer;
pub use nsf::{NSFConfig, NeuralSplineFlow};
pub use spline::{RationalQuadraticSpline, SplineParams};
