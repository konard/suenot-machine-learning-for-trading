//! Training module for DCGAN
//!
//! This module provides:
//! - Training loop implementation
//! - Loss functions (Binary Cross Entropy)
//! - Training configuration and metrics

mod trainer;
mod losses;
mod metrics;

pub use trainer::{Trainer, TrainingConfig};
pub use losses::{generator_loss, discriminator_loss};
pub use metrics::TrainingMetrics;
