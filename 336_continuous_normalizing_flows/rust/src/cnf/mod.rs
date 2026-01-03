//! # Continuous Normalizing Flows Module
//!
//! Implementation of Neural ODE-based Continuous Normalizing Flows.
//!
//! This module provides:
//! - `VelocityField`: Neural network defining the ODE dynamics
//! - `ODESolver`: Numerical ODE solver with trace estimation
//! - `ContinuousNormalizingFlow`: Main CNF model

mod velocity;
mod ode_solver;
mod flow;
mod training;

pub use velocity::VelocityField;
pub use ode_solver::{ODESolver, ODEMethod};
pub use flow::ContinuousNormalizingFlow;
pub use training::{CNFTrainer, TrainingConfig};
