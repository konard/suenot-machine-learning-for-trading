//! Training module for InceptionTime
//!
//! This module provides:
//! - Training loop with early stopping
//! - Loss functions for classification
//! - Evaluation metrics

mod losses;
mod metrics;
mod trainer;

pub use losses::{cross_entropy_loss, focal_loss, weighted_cross_entropy};
pub use metrics::{accuracy, confusion_matrix, f1_score, TrainingMetrics};
pub use trainer::{Trainer, TrainingConfig};
