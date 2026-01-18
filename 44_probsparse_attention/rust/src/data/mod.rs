//! Модуль загрузки и обработки данных
//!
//! - `loader` - Загрузчик данных
//! - `features` - Вычисление признаков
//! - `dataset` - Dataset для обучения

pub mod loader;
pub mod features;
pub mod dataset;

pub use loader::DataLoader;
pub use features::Features;
pub use dataset::TimeSeriesDataset;
