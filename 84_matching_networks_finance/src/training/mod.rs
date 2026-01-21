//! Training utilities for Matching Networks
//!
//! Implements episodic training that simulates few-shot learning scenarios.

mod episodic;

pub use episodic::{EpisodicTrainer, Episode, TrainingConfig, TrainingMetrics};
