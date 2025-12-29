//! Модуль генерации торговых сигналов
//!
//! Содержит стратегии для объединения ценового и sentiment-моментума.

mod divergence;
mod fusion;

pub use divergence::{Divergence, DivergenceDetector, DivergenceType};
pub use fusion::{FusionResult, FusionStrategy};
