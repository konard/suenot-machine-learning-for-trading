//! Energy-Based Models module
//!
//! This module provides various implementations of Energy-Based Models:
//! - Simple neural network energy function
//! - Restricted Boltzmann Machine (RBM)
//! - Score matching training
//! - Online energy estimation

mod energy_net;
mod rbm;
mod score_matching;
mod online;

pub use energy_net::*;
pub use rbm::*;
pub use score_matching::*;
pub use online::*;
