//! Trading strategies module.

mod base;
mod sma_crossover;
mod rsi_strategy;
mod ml_strategy;

pub use base::{Signal, Strategy};
pub use sma_crossover::SmaCrossover;
pub use rsi_strategy::RsiStrategy;
pub use ml_strategy::MlSignalStrategy;
