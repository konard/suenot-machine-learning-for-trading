//! Multi-task training with gradient harmonization
//!
//! This module provides:
//! - Multi-task trainer orchestrating training across all tasks
//! - Gradient harmonization methods (PCGrad, GradNorm)
//! - Task weighting strategies

mod trainer;
mod harmonizer;
mod weighting;

pub use trainer::{MultiTaskTrainer, TrainerConfig};
pub use harmonizer::{GradientHarmonizer, HarmonizerType};
pub use weighting::{TaskWeighter, WeightingStrategy};
