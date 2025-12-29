//! # Модуль обработки данных
//!
//! Подготовка данных для обучения CNN: нормализация, создание окон,
//! формирование батчей.

mod dataset;
mod processor;
mod sample;

pub use dataset::Dataset;
pub use processor::DataProcessor;
pub use sample::Sample;
