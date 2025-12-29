//! # Модуль предобработки данных
//!
//! Подготовка данных для обучения RNN:
//! - Нормализация
//! - Создание последовательностей
//! - Разделение на train/test
//!
//! ## Пример использования
//!
//! ```rust,no_run
//! use crypto_rnn::preprocessing::{DataProcessor, Normalizer};
//! use crypto_rnn::data::Candle;
//!
//! let candles: Vec<Candle> = vec![]; // загруженные свечи
//!
//! // Создаём процессор: 60 шагов назад, 1 шаг вперёд
//! let mut processor = DataProcessor::new(60, 1);
//!
//! // Подготавливаем последовательности
//! let (x, y) = processor.prepare_sequences(&candles).unwrap();
//!
//! // Разделяем на train/test (80%/20%)
//! let (x_train, x_test, y_train, y_test) = processor.train_test_split(&x, &y, 0.8);
//! ```

mod processor;
mod normalizer;
mod features;

pub use processor::DataProcessor;
pub use normalizer::{Normalizer, MinMaxNormalizer, StandardNormalizer};
pub use features::FeatureExtractor;
