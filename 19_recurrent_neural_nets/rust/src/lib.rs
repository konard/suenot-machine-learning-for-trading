//! # Crypto RNN - Рекуррентные нейронные сети для криптовалют
//!
//! Библиотека для прогнозирования цен криптовалют с использованием LSTM/GRU
//! и данных с биржи Bybit.
//!
//! ## Модули
//!
//! - `data` - Получение данных с биржи Bybit
//! - `preprocessing` - Подготовка данных для обучения
//! - `model` - Реализация LSTM и GRU
//! - `utils` - Вспомогательные функции
//!
//! ## Быстрый старт
//!
//! ```rust,no_run
//! use crypto_rnn::data::BybitClient;
//! use crypto_rnn::preprocessing::DataProcessor;
//! use crypto_rnn::model::LSTM;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // 1. Получаем данные
//!     let client = BybitClient::new();
//!     let candles = client.get_klines("BTCUSDT", "1h", 1000).await?;
//!
//!     // 2. Подготавливаем данные
//!     let processor = DataProcessor::new(60, 1); // 60 шагов назад, 1 вперёд
//!     let (x_train, y_train) = processor.prepare_sequences(&candles)?;
//!
//!     // 3. Создаём и обучаем модель
//!     let mut lstm = LSTM::new(5, 64, 1); // 5 признаков, 64 скрытых, 1 выход
//!     lstm.train(&x_train, &y_train, 100, 0.001)?;
//!
//!     Ok(())
//! }
//! ```

pub mod data;
pub mod model;
pub mod preprocessing;
pub mod utils;

// Реэкспорт основных типов для удобства
pub use data::{BybitClient, Candle};
pub use model::{LSTMConfig, LSTM, GRU};
pub use preprocessing::DataProcessor;
