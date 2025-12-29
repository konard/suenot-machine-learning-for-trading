//! Multi-Agent RL algorithms
//!
//! This module provides implementations of various MARL training algorithms.

mod independent;
mod self_play;
mod population;
mod config;

pub use independent::IndependentLearners;
pub use self_play::SelfPlayTrainer;
pub use population::PopulationTrainer;
pub use config::TrainingConfig;
