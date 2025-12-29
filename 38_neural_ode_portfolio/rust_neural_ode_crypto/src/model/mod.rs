//! # Neural ODE Models
//!
//! Neural network models for portfolio dynamics.
//!
//! ## Components
//!
//! - [`network`]: Basic neural network layers
//! - [`portfolio`]: Portfolio-specific ODE dynamics
//! - [`training`]: Training loop and optimization

mod network;
mod portfolio;
mod training;

pub use network::{Layer, MLP, Activation};
pub use portfolio::{
    NeuralODEPortfolio,
    PortfolioState,
    PortfolioDynamics,
    ODEFunc,
};
pub use training::{Trainer, TrainingConfig, LossFunction};
