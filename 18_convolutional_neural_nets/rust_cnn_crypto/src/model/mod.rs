//! # CNN модель для криптовалютного трейдинга
//!
//! Реализация сверточной нейронной сети на базе Burn framework.

mod cnn;
mod config;
mod training;

pub use cnn::CnnModel;
pub use config::{CnnConfig, TrainingConfig};
pub use training::{train_model, evaluate_model};
