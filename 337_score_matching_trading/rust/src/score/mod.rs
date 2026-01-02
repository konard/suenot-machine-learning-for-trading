//! # Score Matching Module
//!
//! Implementation of score matching algorithms for learning data distributions.
//!
//! This module provides:
//! - Score network architecture
//! - Multi-scale denoising score matching
//! - Langevin dynamics for sampling
//! - Training utilities

mod network;
mod training;
mod langevin;

pub use network::{ScoreNetwork, MultiScaleScoreNetwork, Layer, ActivationFn};
pub use training::{DenoisingScoreMatchingTrainer, TrainingConfig, TrainingResult};
pub use langevin::{LangevinDynamics, LangevinConfig};
