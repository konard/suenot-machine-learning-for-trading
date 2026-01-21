//! Training infrastructure for few-shot learning
//!
//! Provides episodic training framework for meta-learning.

mod episode;
mod trainer;

pub use episode::{Episode, EpisodeGenerator, EpisodeConfig};
pub use trainer::{MetaTrainer, MetaTrainerConfig, TrainingResult};
