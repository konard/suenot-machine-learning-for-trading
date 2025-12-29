//! # Rust NLP для криптотрейдинга
//!
//! Библиотека для анализа текстовых данных криптовалютного рынка
//! с использованием Bybit API.
//!
//! ## Модули
//!
//! - `api` - Работа с Bybit API
//! - `nlp` - Обработка естественного языка
//! - `sentiment` - Анализ настроений
//! - `signals` - Генерация торговых сигналов
//! - `models` - Модели данных

pub mod api;
pub mod models;
pub mod nlp;
pub mod sentiment;
pub mod signals;

pub use api::BybitClient;
pub use nlp::{Preprocessor, Tokenizer, Vectorizer};
pub use sentiment::SentimentAnalyzer;
pub use signals::SignalGenerator;
