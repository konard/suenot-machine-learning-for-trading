//! Market impact models and estimation.
//!
//! This module provides:
//! - Almgren-Chriss optimal execution model
//! - Temporary and permanent impact estimation
//! - Impact calibration utilities

mod estimator;
mod models;

pub use estimator::{MarketImpactEstimator, MarketImpactError};
pub use models::{
    AlmgrenChrissModel, AlmgrenChrissParams, ImpactComponent, MarketImpactModel,
    PermanentImpact, TemporaryImpact,
};
