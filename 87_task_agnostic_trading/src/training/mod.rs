//! Multi-task training with gradient harmonization
//!
//! This module provides:
//! - Multi-task trainer for simultaneous optimization of all task heads
//! - Gradient harmonization to balance task losses
//! - Training metrics and progress tracking

mod multi_task;
mod gradient;

pub use multi_task::{MultiTaskTrainer, TrainerConfig, TrainingResult, EpochMetrics, TaskLoss};
pub use gradient::{GradientHarmonizer, HarmonizerConfig, HarmonizationMethod};
