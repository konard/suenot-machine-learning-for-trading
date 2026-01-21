//! Training module for prototypical networks
//!
//! This module provides:
//! - Episodic training framework
//! - Training loop with loss computation
//! - Learning rate scheduling

mod episode;
mod trainer;
mod scheduler;

pub use episode::{Episode, EpisodeGenerator, EpisodeConfig};
pub use trainer::{PrototypicalTrainer, TrainerConfig, TrainingResult};
pub use scheduler::{LearningRateScheduler, SchedulerType};
