//! # Модуль моделей RNN
//!
//! Реализация LSTM и GRU для прогнозирования временных рядов.
//!
//! ## Пример использования
//!
//! ```rust,no_run
//! use crypto_rnn::model::{LSTM, LSTMConfig};
//!
//! // Создаём конфигурацию
//! let config = LSTMConfig::new(5, 64, 1)
//!     .with_learning_rate(0.001)
//!     .with_dropout(0.2);
//!
//! // Создаём модель
//! let mut lstm = LSTM::from_config(config);
//!
//! // Обучаем
//! // lstm.train(&x_train, &y_train, 100, 0.001)?;
//! ```

mod lstm;
mod gru;
mod config;
mod layers;

pub use lstm::LSTM;
pub use gru::GRU;
pub use config::LSTMConfig;
pub use layers::{Dense, Activation};
