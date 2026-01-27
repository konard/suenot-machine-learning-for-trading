//! Training module for transfer learning
//!
//! Provides utilities for:
//! - Pre-training on source domain data
//! - Fine-tuning on target domain data
//! - Domain adaptation during training

mod trainer;

pub use trainer::{TransferTrainer, TrainingConfig, AdaptationConfig, TrainingResult};
