//! Модуль для загрузки и обработки мультивариантных данных
//!
//! Предоставляет инструменты для подготовки данных для Stockformer,
//! включая загрузку, feature engineering и создание датасетов.

mod loader;
mod features;
mod dataset;

pub use loader::MultiAssetLoader;
pub use features::{Features, FeatureConfig, calculate_features};
pub use dataset::{MultiAssetDataset, CorrelationMatrix, DataSplit};
